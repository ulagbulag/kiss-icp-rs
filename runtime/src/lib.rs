use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use rayon::ThreadPoolBuilder;

static IS_INITED: AtomicBool = AtomicBool::new(false);

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum SystemType {
    /// Use all threads without the main thread
    #[default]
    Generic,
    /// Use all threads even with the main thread
    Python,
}

pub fn init(system_type: SystemType) -> Result<()> {
    if !IS_INITED.swap(true, Ordering::SeqCst) {
        let threads = prepare_threads()?;

        let mut builder = ThreadPoolBuilder::new().num_threads(threads.len());
        if matches!(system_type, SystemType::Python) {
            builder = builder.use_current_thread();
        }
        builder.build_global()?;

        bind_threads(threads)?;
    }
    Ok(())
}

#[cfg(not(feature = "numa"))]
const fn get_topology() -> Result<()> {
    Ok(())
}

#[cfg(feature = "numa")]
fn get_topology() -> Result<::hwlocality::Topology> {
    ::hwlocality::Topology::new().map_err(Into::into)
}

#[cfg(not(feature = "numa"))]
fn prepare_threads() -> Result<Vec<usize>> {
    use std::thread;

    // heuristic values (Feb 03, 2024)
    const MAX_THREADS: usize = 32;

    let num_threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(MAX_THREADS);
    Ok((0..num_threads).collect())
}

#[cfg(feature = "numa")]
fn prepare_threads() -> Result<Vec<usize>> {
    use rand::{
        distributions::{Distribution, Uniform},
        thread_rng,
    };

    // get NUMA/CPUs info
    let topology = get_topology()?;
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
    let cpu_begin = numa_node * num_threads_per_cpu;
    let cpu_end = cpu_begin + num_threads_per_cpu;
    Ok((cpu_begin..cpu_end).collect())
}

#[cfg(not(feature = "numa"))]
#[inline]
fn bind_threads(_: Vec<usize>) -> Result<()> {
    Ok(())
}

#[cfg(feature = "numa")]
fn bind_threads(threads: Vec<usize>) -> Result<()> {
    use hwlocality::cpu::{binding::CpuBindingFlags, cpuset::CpuSet};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    threads.into_par_iter().try_for_each(|idx| {
        // bind the given thread into the NUMA node
        let topology = get_topology()?;
        let cpus = {
            let mut res = CpuSet::new();
            res.set(idx);
            res
        };
        topology.bind_cpu(&cpus, CpuBindingFlags::THREAD)?;
        Ok(())
    })
}
