# Tor V3 vanity address generator

Usage:
```
./vanity_torv3_rocm-(ARCH) [-i] [-d N] pattern1 [pattern_2] [pattern_3] ... [pattern_n]
```

`-i` will display the keygen rate every 20 seconds in million addresses per second.
`-d` use ROCm device with index N (counting from 0). This argument can be repeated multiple times with different N.

Example:
```
./vanity_torv3_rocm-gfx1030 -i 4?xxxxx 533333 655555 777777 p99999
```
Capture generated keys simply by output redirection into a file :
`$ ./vanity_torv3_cuda test | tee -a keys.txt`

You can then use the `genpubs.py` scripts under `util/` to generate all the tor secret files :
 `$ python3 genpubs.py keys.txt`
 
A folder called `generated-<timestamp>` will be generated

## Performance

This generator can check ~39.3 million keys/second on a single Radeon RX 6700 XT. (Core i7-4770S+mkp224o=14.6M kps)
Multiple patterns don't slow down the search.
Max pattern prefix search length is 32 characters to allow offset searching ie. `????wink??????test???`
Anything beyond 12 characters will probably take a few hundred years...
Only these characters are permitted in an address:
```
abcdefghijklmnopqrstuvwxyz234567
```

## Build instructions

```
make ARCH=gfx1030 EXPAND=100
```
