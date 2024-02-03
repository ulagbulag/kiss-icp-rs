# KISS-ICP

An unofficial pure Rust implementation of [KISS-ICP](https://github.com/PRBonn/kiss-icp.git), a LiDAR Odometry pipeline that just works on most of the cases without tunning any parameter.

## Features

- Hardware Topology-aware Parallel Computing support (`numa` feature, enabled by default)
- [KISS-ICP python](https://pypi.org/project/kiss-icp/) plugin Integration (Seamless **CMake** supported is planned)

## Install

### Python (kiss-icp)

TBD

### Rust Library

```sh
cargo add kiss-icp
```

## LICENSE

If a separate `LICENSE` file is provided among the Rust packages, it will comply with that license and not the "Main License" in the root path of this repository.
All other packages follow the MIT license, the "Main License" of this repository.
