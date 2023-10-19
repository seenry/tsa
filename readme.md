# How 2 login
ssh -i `.pem file` ubuntu@3.16.162.115

# Running Stuff
Code is in the `tsa` folder (`/home/ubuntu/tsa`)
To run, cd into `tsa` and do `mpirun -np 4 ./bin`

This should generate three csv files: `p2p_uni`, `p2p_bi`, and `all_gather`.

# Todo:
Clean up `GPUNetwork::Print`
