time python main.py --dataset mnist --dp_mechanism no_dp --dp_epsilon 100 --dp_clip 50 --gpu 0 --frac 0.2 --num_users 100 --epochs 100 --runs 10
time python main.py --dataset mnist --dp_mechanism no_dp --dp_epsilon 100 --dp_clip 50 --gpu 0 --frac 0.2 --num_users 50 --epochs 100 --runs 10
time python main.py --dataset mnist --dp_mechanism no_dp --dp_epsilon 100 --dp_clip 50 --gpu 0 --frac 0.2 --num_users 200 --epochs 100 --runs 10

time python main.py --dataset mnist --dp_mechanism no_dp --dp_epsilon 100 --dp_clip 50 --gpu 0 --frac 0.1 --num_users 100 --epochs 100 --runs 10
time python main.py --dataset mnist --dp_mechanism no_dp --dp_epsilon 100 --dp_clip 50 --gpu 0 --frac 0.05 --num_users 100 --epochs 100 --runs 10
time python main.py --dataset mnist --dp_mechanism no_dp --dp_epsilon 100 --dp_clip 50 --gpu 0 --frac 0.5 --num_users 100 --epochs 100 --runs 10

time python main.py --dataset mnist --dp_mechanism no_dp --dp_epsilon 100 --dp_clip 50 --gpu 0 --frac 0.2 --num_users 100 --epochs 100 --runs 10 --minibatch 150
time python main.py --dataset mnist --dp_mechanism no_dp --dp_epsilon 100 --dp_clip 50 --gpu 0 --frac 0.2 --num_users 100 --epochs 100 --runs 10 --minibatch 300