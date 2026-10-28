from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.double_dqn import DoubleDQNAgent
from agents.dueling_double_dqn import DuelingDoubleDQNAgent
from agents.gae_actor_critic import GAEActorCritic
from replay import ReplayBuffer
import numpy as np
import torch
import math

import pandas as pd
import os
from datetime import datetime

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 50, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 1000, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('net_file', 'nets/2way-double-intersection/double-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-double-intersection/double-intersection.rou.xml', '')
flags.DEFINE_bool('use_gui', True, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 1, '')
flags.DEFINE_string('network', 'double_dqn', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_float('eps_start', 1.0, '')
flags.DEFINE_float('eps_end', 0.1, '')
flags.DEFINE_integer('eps_decay', 83000, '')
flags.DEFINE_integer('target_update', 3000, '')
flags.DEFINE_string('network_file', '', '')
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_bool('multi_agent', False, 'control all traffic lights in the network')
flags.DEFINE_string('results_dir', 'results', 'directory to save results')

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_metrics(metrics, filename):
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(FLAGS.results_dir, filename), index=False)
    print(f"Metrics saved to {os.path.join(FLAGS.results_dir, filename)}")

def run_single_agent_training(env, agent, replay_buffer):
    metrics = {
        'episode': [],
        'rewards': [],
        'speeds': [],
        'queue_lengths': [],
        'waiting_times': []
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{FLAGS.network}_metrics_{timestamp}.csv"

    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False
        
        ep_rewards = []
        ep_speeds = []
        ep_queues = []
        ep_waiting = []
        
        while not done:
            state = env.compute_state
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)
            next_state, reward, done, info = env.step(action)
            
            if info['do_action'] is None or next_state is None:
                invalid_action = True
                continue
            invalid_action = False
            
            replay_buffer.add(env.train_state, next_state, reward, info['do_action'], done)
            
            if FLAGS.mode == 'train':
                agent.optimize_model()
                agent.update_target_network(replay_buffer.steps_done)
            
            ep_rewards.append(reward)
            # Collect metrics from traffic signal
            ts = env.traffic_signal
            ep_speeds.append(ts.get_average_speed())
            ep_queues.append(ts.get_average_queue_length())
            ep_waiting.append(ts.get_average_waiting_time())

        env.close()
        
        avg_reward = np.mean(ep_rewards)
        avg_speed = np.mean(ep_speeds)
        avg_queue = np.mean(ep_queues)
        avg_waiting = np.mean(ep_waiting)
        
        metrics['episode'].append(episode + 1)
        metrics['rewards'].append(avg_reward)
        metrics['speeds'].append(avg_speed)
        metrics['queue_lengths'].append(avg_queue)
        metrics['waiting_times'].append(avg_waiting)
        
        print(f'Episode {episode+1}/{FLAGS.num_episodes} - Reward: {avg_reward:.4f}, Speed: {avg_speed:.2f}, Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}')
        
    save_metrics(metrics, filename)


def run_multi_agent_training(env, agent, replay_buffer):
    num_agents = env.num_agents
    metrics = {
        'episode': [],
        'rewards': [],
        'speeds': [],
        'queue_lengths': [],
        'waiting_times': []
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{FLAGS.network}_metrics_{timestamp}.csv"

    for episode in range(FLAGS.num_episodes):
        states = env.reset()
        done = False
        invalid_actions = [False] * num_agents
        
        ep_rewards = []
        ep_speeds = []
        ep_queues = []
        ep_waiting = []
        
        while not done:
            states = env._compute_states()
            actions = agent.select_actions(states, agent.steps_done, invalid_actions)
            next_states, rewards, done, info = env.step(actions)

            for i in range(num_agents):
                if info['do_actions'][i] is None:
                    invalid_actions[i] = True
                else:
                    invalid_actions[i] = False
                if info['do_actions'][i] is not None and next_states[i] is not None:
                    all_states = [np.asarray(s, dtype=np.float32) for s in states]
                    all_next = [
                        np.asarray(next_states[j], dtype=np.float32)
                        if next_states[j] is not None else all_states[j]
                        for j in range(num_agents)
                    ]
                    replay_buffer.add(
                        i, states[i], info['do_actions'][i], next_states[i],
                        rewards[i], done, all_states, all_next,
                    )
            
            if FLAGS.mode == 'train' and hasattr(agent, 'store_transitions'):
                agent.store_transitions(states, info['do_actions'], next_states, rewards, [done]*num_agents)
            
            if FLAGS.mode == 'train':
                agent.optimize_from_buffer(replay_buffer)
            
            ep_rewards.append(np.mean(rewards))
            
            # Aggregate metrics from all agents
            speeds = [ts.get_average_speed() for ts in env.traffic_signals]
            queues = [ts.get_average_queue_length() for ts in env.traffic_signals]
            waiting = [ts.get_average_waiting_time() for ts in env.traffic_signals]
            
            ep_speeds.append(np.mean(speeds))
            ep_queues.append(np.mean(queues))
            ep_waiting.append(np.mean(waiting))

        env.close()
        
        avg_reward = np.mean(ep_rewards)
        avg_speed = np.mean(ep_speeds)
        avg_queue = np.mean(ep_queues)
        avg_waiting = np.mean(ep_waiting)
        
        metrics['episode'].append(episode + 1)
        metrics['rewards'].append(avg_reward)
        metrics['speeds'].append(avg_speed)
        metrics['queue_lengths'].append(avg_queue)
        metrics['waiting_times'].append(avg_waiting)
        
        print(f'Episode {episode+1}/{FLAGS.num_episodes} - Reward: {avg_reward:.4f}, Speed: {avg_speed:.2f}, Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}')

    save_metrics(metrics, filename)


def main(argv):
    del argv
    env = SumoEnv(
        net_file=FLAGS.net_file,
        route_file=FLAGS.route_file,
        skip_range=FLAGS.skip_range,
        simulation_time=FLAGS.simulation_time,
        yellow_time=FLAGS.yellow_time,
        delta_rs_update_time=FLAGS.delta_rs_update_time,
        use_gui=FLAGS.use_gui,
        multi_agent=FLAGS.multi_agent,
    )

    env.reset()

    replay_buffer = ReplayBuffer(capacity=20000)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    if FLAGS.network == 'double_dqn':
        agent = DoubleDQNAgent(
            FLAGS.mode, replay_buffer, FLAGS.target_update, FLAGS.gamma,
            FLAGS.eps_start, FLAGS.eps_end, FLAGS.eps_decay,
            input_dim, output_dim, FLAGS.batch_size, FLAGS.network_file,
        )
    elif FLAGS.network == 'dueling_ddqn':
        agent = DuelingDoubleDQNAgent(
            FLAGS.mode, replay_buffer, FLAGS.target_update, FLAGS.gamma,
            FLAGS.eps_start, FLAGS.eps_end, FLAGS.eps_decay,
            input_dim, output_dim, FLAGS.batch_size, FLAGS.network_file,
        )
    elif FLAGS.network == 'shared_policy_gae_actor_critic':
        agent = GAEActorCritic(
            input_dim, output_dim, FLAGS.mode, FLAGS.gamma
        )
    else:
        raise ValueError(f"Unknown network type: {FLAGS.network}")

    run_single_agent_training(env, agent, replay_buffer)


if __name__ == '__main__':
    app.run(main)
