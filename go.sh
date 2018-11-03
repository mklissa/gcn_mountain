

#!/usr/bin/bash

seed_bias=180



if [ $1 = 1 ]; then
	name="yes_graph"
else
	name="no_graph"
fi

tmux new-session -d -s $name

for ((i=0;i<10;i++));do

	let seed=seed_bias+i

    tmux new-window -t $name: -n "seed$seed" 
    tmux send-keys -t $name: "python car.py --gen $1 --seed $seed"
    tmux send-keys -t $name: Enter


done



tmux -2 attach-session -t $name
