extern crate kiss_icp_ops as ops;

pub mod deskew;
pub mod metrics;
pub mod preprocessing;
pub mod threshold;
pub mod types;
pub mod voxel_hash_map;

pub mod runtime {
    use std::sync::atomic::{AtomicBool, Ordering};

    use anyhow::Result;
    use rayon::ThreadPoolBuilder;

    static IS_INITED: AtomicBool = AtomicBool::new(false);

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub enum SystemType {
        #[default]
        Executable,
        Library,
    }

    pub fn init(system_type: SystemType) -> Result<()> {
        if !IS_INITED.swap(true, Ordering::SeqCst) {
            let mut builder = ThreadPoolBuilder::new().num_threads(prepare_threads()?);
            if matches!(system_type, SystemType::Library) {
                builder = builder.use_current_thread();
            }
            builder.build_global()?;
        }
        Ok(())
    }

    #[cfg(not(feature = "numa"))]
    #[inline]
    fn prepare_threads() -> Result<usize> {
        use std::thread;

        // heuristic values (Feb 03, 2024)
        const MAX_THREADS: usize = 32;

        Ok(thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_THREADS))
    }

    #[cfg(feature = "numa")]
    #[inline]
    fn prepare_threads() -> Result<usize> {
        // select a NUMA node
        let topology = ::hwlocality::Topology::new()?;
        select_numa_node(&topology)
    }

    #[cfg(feature = "numa")]
    #[inline]
    fn select_numa_node(topology: &::hwlocality::Topology) -> Result<usize> {
        use hwlocality::cpu::{binding::CpuBindingFlags, cpuset::CpuSet};
        use rand::{
            distributions::{Distribution, Uniform},
            thread_rng,
        };

        // get NUMA/CPUs info
        let all_numa_nodes = topology.nodeset();
        let all_cpus = topology.cpuset();

        // count the resources
        let num_numa_nodes = all_numa_nodes
            .last_set()
            .map(|set| set.into())
            .unwrap_or(0usize)
            + 1;
        let num_cpus = all_cpus.last_set().map(|set| set.into()).unwrap_or(0usize) + 1;
        let num_threads_per_cpu = num_cpus / num_numa_nodes;

        // pick a random NUMA node
        let numa_node = Uniform::new(0usize, num_numa_nodes).sample(&mut thread_rng());

        // get all the CPUs in the NUMA node
        let cpus = {
            let cpu_begin = numa_node * num_threads_per_cpu;
            let cpu_end = cpu_begin + num_threads_per_cpu;

            let mut res = CpuSet::new();
            for idx in cpu_begin..cpu_end {
                res.set(idx);
            }
            res
        };

        // bind the process into the NUMA node
        topology.bind_cpu(&cpus, CpuBindingFlags::PROCESS)?;

        // return the count of available threads
        Ok(num_threads_per_cpu)
    }
}
