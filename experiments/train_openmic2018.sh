#!/bin/bash

# Runs all experiments listed at the bottom. Each experiment consists of a
# given number of repetitions using a particular base name for the weights and
# predictions files. Each single repetition checks if it was already run or is
# currently being run, creates a lockfile, trains the network, computes the
# predictions, and removes the lockfile. To distribute runs between multiple
# GPUs, just run this script multiple times with different --cuda-device=N.

here="${0%/*}"
outdir="$here/../results/openmic2018"
logdir="$here/../logs/openmic2018"

train_if_free() {
	modelfile="$1"
	echo "$modelfile"
	logsubdir="$logdir/${modelfile%.*}"
	modelfile="$outdir/$modelfile"
	mkdir -p "${modelfile%/*}"
	if [ ! -f "$modelfile" ] && [ ! -f "$modelfile.lock" ]; then
		for gpu in "$@" ''; do [[ "$gpu" == "--cuda-device="? ]] && break; done
		echo "$HOSTNAME: $gpu" > "$modelfile.lock"
		$PYTHON_COMMAND "$here"/../train.py "$modelfile" --logdir="$logsubdir" "${@:2}" #&& \
			#$PYTHON_COMMAND "$here"/../predict.py "$modelfile" "${modelfile%.*}.preds" --var batchsize=1 $gpu
		rm "$modelfile.lock"
	fi
}

train() {
	repeats="$1"
	name="$2"
	for (( r=1; r<=$repeats; r++ )); do
		train_if_free "$name"_r$r.mdl "${@:3}"
	done
}

data="--var dataset=openmic2018"
model=
metrics=
training=
train 1 v2/default $data $model $metrics $training "$@"

data="--var dataset=openmic2018 --var data.audio_dir=audio/000"
model=
metrics=
training="--var train.epochs=20 --var train.epochsize=100"
train 1 v2/overfit-test $data $model $metrics $training "$@"