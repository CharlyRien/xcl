//! LOCAL (agent V): what the off-game replay binary (`examples/lreplay.rs`) needs from the crate
//! and cannot reach from outside it. The replay lives outside the library so that editing it
//! does not recompile XCL.

use windows::Win32::Graphics::Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC;

use crate::gpu::recomp;

/// The translation of a console shader as the title's run makes it (HLSL form, never the compact
/// form): the HLSL, its DXC profile and its static samplers.
pub fn console_translation(blob: &[u8], stage: recomp::pso::Stage, sampler_base: u32, windows: u32, split_wave64: bool) -> Result<(String, &'static str, Vec<recomp::pso::StaticSampler>), String> {
    recomp::pso::console_translation(blob, stage, sampler_base, windows, split_wave64, recomp::CompactCs::Never).map(|(t, profile, _)| (t.hlsl, profile, t.static_samplers))
}

/// `real::pc_topology_type`
pub fn pc_topology_type(g: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC) {
    super::real::pc_topology_type(g)
}
