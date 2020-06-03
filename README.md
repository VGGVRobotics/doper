# DOPEr: dope af

## Example commands
Start training, jax on cpu
```buildoutcfg
JAX_PLATFORM_NAME=cpu python3.8 ./scripts/train.py -c ./configs/jax_ball.yaml
```
Start training, jax on gpu
```buildoutcfg
JAX_PLATFORM_NAME=gpu python3.8 ./scripts/train.py -c ./configs/jax_ball.yaml
```

The scripts and notebooks are in `./scripts` folder

No tests yet

Sample maps are in the `assets` folder. We use `.svg` vector maps
