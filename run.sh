#!/bin/bash

if [ -z "$1" ]
  then echo "Please provide the name of the game, e.g.  ./run_cpu breakout "; exit 0
fi
ENV=$1

# FRAMEWORK OPTIONS
FRAMEWORK="neswrap" # Wrapper for the FCEUX Nintendo emulator.
game_path=$PWD"/roms/"
env_params="useRGB=true"
steps=50000000 # Total steps to run the model.
save_freq=30000 # Save every save_freq steps. Save early and often! 125k for Atari.

# PREPROCESSOR OPTIONS
preproc_net="\"net_downsample_2x_full_y\""
pool_frms_type="\"max\""
pool_frms_size=1 # Changed from 2 for Atari, since we don't have the same limitations for NES.
initial_priority="false"
state_dim=7056 # The number of pixels in the screen.
ncols=1 # Represents just the Y (ie - grayscale) channel.

# AGENT OPTIONS
agent="NeuralQLearner"
agent_type="DQN3_0_1"
agent_name=$agent_type"_"$1"_FULL_Y"
actrep=8 # Number of times an action is repeated (and a screen returned). 4 for Atari...
ep=1 # The probability of choosing a random action rather than the best predicted action.
eps_end=0.1 # What epsilon ends up as going forward.
eps_endt=1000000 # This probability decreases over time, presumably as we get better.
max_reward=10000 # Rewards are clipped to this value.
min_reward=-10000 # Ditto.
rescale_r=1 # Rescale rewards to [0, 1]
gameOverPenalty=0 # Gives a negative reward equal to the current score upon dying.

# LEARNING OPTIONS
lr=0.001 # .00025 for Atari.
learn_start=5000 # Only start learning after this many steps. Should be bigger than bufferSize. Was set to 50k for Atari.
replay_memory=1000000 # Set small to speed up debugging. 10M is the Atari setting... Big memory object!
n_replay=4 # Minibatches to learn from each learning step.
nonEventProb=nil # Probability of selecting a non-reward-bearing experience.
clip_delta=1 # Limit the delta to +/- 1.

# Q NETWORK OPTIONS
netfile="\"convnet_nes\""
target_q=10000 # How many steps to replace the target Q nework with the updated one. 10k for Atari...
update_freq=4 # How often do we update the Q network? 
hist_len=4 # Number of trailing frames to input into the Q network. 4 for Atari...
discount=0.99 # Discount rate given to future rewards.

# VALIDATION AND EVALUATION
eval_freq=40000 # Evaluate the model every eval_freq steps by calculating the score per episode for a few games. 250k for Atari.
eval_steps=20000 # How many steps does an evaluation last? 125k for Atari.
prog_freq=40000 # How often do you want a progress report?

# PERFORMANCE AND DEBUG OPTIONS
gpu=0 # Zero means "use the GPU" which is a bit confusing... -1 for CPU.
num_threads=8
verbose=3 # 2 is default. 3 turns on debugging messages about what the model is doing.
random_starts=0 # How many NOOPs to perform at the start of a game (random number up to this value). Shouldn't matter for SMB?
seed=1
#saved_network="/home/ehrenbrav/projects/DeepQNetwork/dqn/short_memory.t7"

# THE UGLY UNDERBELLY
pool_frms="type="$pool_frms_type",size="$pool_frms_size

agent_params="lr="$lr",ep="$ep",ep_end="$eps_end",ep_endt="$eps_endt",discount="$discount",hist_len="$hist_len",learn_start="$learn_start",replay_memory="$replay_memory",update_freq="$update_freq",n_replay="$n_replay",network="$netfile",preproc="$preproc_net",state_dim="$state_dim",minibatch_size=256,ncols="$ncols",bufferSize=1024,valid_size=1000,target_q="$target_q",clip_delta="$clip_delta"",min_reward="$min_reward",max_reward="$max_reward",rescale_r="$rescale_r",nonEventProb="$nonEventProb" 

args="-framework $FRAMEWORK -game_path $game_path -name $agent_name -env $ENV -env_params $env_params -agent $agent -agent_params $agent_params -steps $steps -eval_freq $eval_freq -eval_steps $eval_steps -prog_freq $prog_freq -save_freq $save_freq -actrep $actrep -gpu $gpu -random_starts $random_starts -pool_frms $pool_frms -seed $seed -threads $num_threads -verbose $verbose -gameOverPenalty $gameOverPenalty" #-network $saved_network"

# Copy stdout and stderr to a logfile.
LOGFILE="logs/dqn_log_`/bin/date +\"%F:%R\"`"
exec > >(tee -i ${LOGFILE})
exec 2>&1

echo $args

cd dqn
../torch/bin/qlua train_agent.lua $args
