use std::collections::{HashMap, HashSet};
use std::time::Instant;

use impeller2::types::ComponentId;

use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::profile::{Profiler, TickTimings};
use crate::world::World;

type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

pub struct CraneliftExec {
    pub metadata: ExecMetadata,
    tick_fn: TickFn,
    input_ids: Vec<ComponentId>,
    output_ids: Vec<ComponentId>,
    output_keep_mask: Vec<bool>,
    mutable_overlap: Vec<(usize, usize)>,
    output_buffers: Vec<Vec<u8>>,
    dup_scratch: Vec<u8>,
    _compiled: cranelift_mlir::lower::CompiledModule,
}

// The JIT-compiled function pointer is safe to send between threads;
// cranelift_jit::JITModule is not inherently Send/Sync but we only
// call the function pointer from a single thread at a time.
unsafe impl Send for CraneliftExec {}
unsafe impl Sync for CraneliftExec {}

impl CraneliftExec {
    pub fn new(
        metadata: ExecMetadata,
        compiled: cranelift_mlir::lower::CompiledModule,
        world: &World,
    ) -> Result<Self, Error> {
        let fn_ptr = compiled.get_main_fn();
        let tick_fn: TickFn = unsafe { std::mem::transmute(fn_ptr) };

        let mut input_ids = Vec::new();
        let mut seen_inputs = HashSet::new();
        for slot in &metadata.arg_slots {
            if seen_inputs.insert(slot.component_id) {
                input_ids.push(slot.component_id);
            }
        }

        let mut output_ids = Vec::new();
        let mut seen_outputs = HashSet::new();
        let mut output_keep_mask = Vec::with_capacity(metadata.ret_ids.len());
        for id in &metadata.ret_ids {
            let is_new = seen_outputs.insert(*id);
            output_keep_mask.push(is_new);
            if is_new {
                output_ids.push(*id);
            }
        }

        let mut output_slot_by_id = HashMap::new();
        for (slot, id) in output_ids.iter().enumerate() {
            output_slot_by_id.insert(*id, slot);
        }
        let mutable_overlap: Vec<(usize, usize)> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(input_slot, id)| {
                output_slot_by_id
                    .get(id)
                    .copied()
                    .map(|output_slot| (input_slot, output_slot))
            })
            .collect();

        let mut output_buffers = Vec::new();
        let mut max_buf_size = 0usize;
        for id in &output_ids {
            let col = world.column_by_id(*id).ok_or(Error::ComponentNotFound)?;
            max_buf_size = max_buf_size.max(col.column.len());
            output_buffers.push(vec![0u8; col.column.len()]);
        }
        let dup_scratch = vec![0u8; max_buf_size.max(1024)];

        Ok(Self {
            metadata,
            tick_fn,
            input_ids,
            output_ids,
            output_keep_mask,
            mutable_overlap,
            output_buffers,
            dup_scratch,
            _compiled: compiled,
        })
    }

    pub fn invoke_batch(
        &mut self,
        world: &mut World,
        n: u64,
        _detailed: bool,
    ) -> Result<TickTimings, Error> {
        let batch_ticks = n.max(1) as usize;

        for batch_idx in 0..batch_ticks {
            let input_ptrs: Vec<*const u8> = self
                .input_ids
                .iter()
                .map(|id| {
                    world
                        .column_by_id(*id)
                        .map(|col| col.column.as_ptr())
                        .unwrap_or(std::ptr::null())
                })
                .collect();

            let mut dedup_idx = 0usize;
            let mut output_ptrs: Vec<*mut u8> = Vec::with_capacity(self.metadata.ret_ids.len());
            for keep in &self.output_keep_mask {
                if *keep {
                    output_ptrs.push(self.output_buffers[dedup_idx].as_mut_ptr());
                    dedup_idx += 1;
                } else {
                    output_ptrs.push(self.dup_scratch.as_mut_ptr());
                }
            }

            unsafe {
                (self.tick_fn)(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr());
            }

            if batch_idx + 1 < batch_ticks {
                for &(input_slot, output_slot) in &self.mutable_overlap {
                    let id = self.input_ids[input_slot];
                    if let Some(host) = world.host.get_mut(&id) {
                        let src = &self.output_buffers[output_slot];
                        let len = host.buffer.len().min(src.len());
                        host.buffer[..len].copy_from_slice(&src[..len]);
                    }
                }
            }
        }

        for (slot, id) in self.output_ids.iter().enumerate() {
            let host = world.host.get_mut(id).ok_or(Error::ComponentNotFound)?;
            let src = &self.output_buffers[slot];
            let len = host.buffer.len().min(src.len());
            host.buffer[..len].copy_from_slice(&src[..len]);
        }

        Ok(TickTimings::default())
    }
}

pub struct CraneliftWorldExec {
    pub world: World,
    pub tick_exec: CraneliftExec,
    pub startup_exec: Option<CraneliftExec>,
    pub profiler: Profiler,
}

impl CraneliftWorldExec {
    pub fn new(
        world: World,
        tick_exec: CraneliftExec,
        startup_exec: Option<CraneliftExec>,
    ) -> Self {
        Self {
            world,
            tick_exec,
            startup_exec,
            profiler: Default::default(),
        }
    }

    pub fn run(&mut self) -> Result<(), Error> {
        let start = &mut Instant::now();
        let ticks_per_telemetry = self.world.ticks_per_telemetry();

        let tick_start = Instant::now();
        self.tick_exec.invoke_batch(
            &mut self.world,
            ticks_per_telemetry,
            self.profiler.detailed_timing,
        )?;
        let tick_elapsed = tick_start.elapsed();
        self.profiler.execute_buffers.observe_duration(tick_elapsed);

        *start = Instant::now();
        for _ in 0..ticks_per_telemetry {
            self.world.advance_tick();
        }
        self.profiler.add_to_history.observe(start);
        Ok(())
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.profiler.profile(
            self.world.sim_time_step().0,
            self.world.ticks_per_telemetry(),
        )
    }
}
