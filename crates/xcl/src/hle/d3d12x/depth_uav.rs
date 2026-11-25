//! Depth-stencil + unordered-access color twins (`ALLOW_DEPTH_STENCIL | ALLOW_UNORDERED_ACCESS`).
//!
//! Console titles bind depth-stencil buffers as UAVs in compute passes (for example, writing
//! tiled material classifications into depth and bitmasks into stencil, then testing them with
//! `DEPTH_READ` / `EQUAL` draws). PC D3D12 rejects `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`
//! on depth-stencil textures, so `device_x::pc_resource_desc` strips bit 2 (`ALLOW_UNORDERED_ACCESS`).
//! Without a writable host twin, every UAV of the depth-stencil texture becomes a null view and
//! compute writes vanish.
//!
//! This module pairs each single-sample depth+UAV texture with lazily allocated `ALLOW_UNORDERED_ACCESS`
//! color twin texture(s) (`R32_TYPELESS` / `R16_TYPELESS` for the depth plane and `R8_TYPELESS` for
//! the stencil plane on two-plane formats). Outside synchronization ops, every twin stays in
//! `D3D12_RESOURCE_STATE_UNORDERED_ACCESS` (valid on both DIRECT and COMPUTE queues). On DIRECT
//! command list transitions out of / into `UNORDERED_ACCESS` (or before `OMSetRenderTargets` binds
//! the DSV), dirty planes synchronize lazily:
//! * Single-plane `R32_TYPELESS` / `R16_TYPELESS`: direct `CopyResource`.
//! * Two-plane `R32G8X24_TYPELESS` depth (plane 0) and any two-plane stencil (plane 1):
//!   per-subresource buffer-through `CopyTextureRegion` (exact bit-for-bit transfer).
//! * Two-plane `R24G8_TYPELESS` depth (plane 0, e.g. console `D16_UNORM_S8_UINT` / `D24_UNORM_S8_UINT`):
//!   fullscreen depth-write pass (`SV_Depth`, stencil writes disabled) for twin -> depth and a
//!   compute copy (`R24_UNORM_X8_TYPELESS` -> `R16_UNORM`/`R32_FLOAT`) for depth -> twin,
//!   restoring all list graphics/descriptor bindings afterward.
//!
//! Set `XCL_DEPTH_UAV_TWIN=0` to disable color twins.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::{Interface, PCSTR};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::list_x;
use super::real;

#[cfg(test)]
const STATE_COMMON: u32 = 0;
const STATE_UAV: u32 = 0x8;
const STATE_DEPTH_WRITE: u32 = 0x10;
const STATE_COPY_DEST: u32 = 0x400;
const STATE_COPY_SOURCE: u32 = 0x800;
const STATE_SRV_ALL: u32 = 0x40 | 0x80;

const FLAG_DEPTH_STENCIL: u32 = 0x2;
const FLAG_UAV: u32 = 0x4;

/// How a depth resource's plane 0 synchronizes with its `depth_twin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DepthSyncKind {
    /// Single-plane `R32_TYPELESS` or `R16_TYPELESS`: `CopyResource` directly.
    DirectCopy,
    /// Two-plane `R32G8X24_TYPELESS` plane 0 (`R32_TYPELESS`): buffer-through `CopyTextureRegion`.
    BufferThrough,
    /// Two-plane `R24G8_TYPELESS` plane 0: fullscreen depth-write pass (twin -> depth) and CS copy (depth -> twin).
    FullscreenPass { twin_srv_fmt: i32, twin_uav_fmt: i32 },
}

/// Twin allocation & sync plan derived from a console resource format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TwinPlan {
    pub depth_twin_fmt: i32,
    pub stencil_twin_fmt: Option<i32>,
    pub depth_sync: DepthSyncKind,
    pub two_plane: bool,
}

/// Choose the color twin format(s) and sync strategy for a console depth-stencil format.
pub(super) fn plan_for_console_format(console_fmt: u32) -> Option<TwinPlan> {
    match console_fmt {
        // D32_FLOAT / R32_TYPELESS
        39 | 40 => Some(TwinPlan {
            depth_twin_fmt: DXGI_FORMAT_R32_TYPELESS.0,
            stencil_twin_fmt: None,
            depth_sync: DepthSyncKind::DirectCopy,
            two_plane: false,
        }),
        // D16_UNORM / R16_TYPELESS
        53 | 55 => Some(TwinPlan {
            depth_twin_fmt: DXGI_FORMAT_R16_TYPELESS.0,
            stencil_twin_fmt: None,
            depth_sync: DepthSyncKind::DirectCopy,
            two_plane: false,
        }),
        // D32_FLOAT_S8X24_UINT / R32G8X24_TYPELESS
        19 | 20 => Some(TwinPlan {
            depth_twin_fmt: DXGI_FORMAT_R32_TYPELESS.0,
            stencil_twin_fmt: Some(DXGI_FORMAT_R8_TYPELESS.0),
            depth_sync: DepthSyncKind::BufferThrough,
            two_plane: true,
        }),
        // Console D16_UNORM_S8_UINT family -> PC R24G8_TYPELESS
        118..=120 => Some(TwinPlan {
            depth_twin_fmt: DXGI_FORMAT_R16_TYPELESS.0,
            stencil_twin_fmt: Some(DXGI_FORMAT_R8_TYPELESS.0),
            depth_sync: DepthSyncKind::FullscreenPass {
                twin_srv_fmt: DXGI_FORMAT_R16_UNORM.0,
                twin_uav_fmt: DXGI_FORMAT_R16_UNORM.0,
            },
            two_plane: true,
        }),
        // D24_UNORM_S8_UINT / R24G8_TYPELESS
        44 | 45 => Some(TwinPlan {
            depth_twin_fmt: DXGI_FORMAT_R32_TYPELESS.0,
            stencil_twin_fmt: Some(DXGI_FORMAT_R8_TYPELESS.0),
            depth_sync: DepthSyncKind::FullscreenPass {
                twin_srv_fmt: DXGI_FORMAT_R32_FLOAT.0,
                twin_uav_fmt: DXGI_FORMAT_R32_FLOAT.0,
            },
            two_plane: true,
        }),
        _ => None,
    }
}

/// Which plane(s) a view or barrier targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PlaneMask {
    Depth,
    Stencil,
    Both,
}

impl PlaneMask {
    fn includes_depth(self) -> bool {
        matches!(self, PlaneMask::Depth | PlaneMask::Both)
    }

    fn includes_stencil(self) -> bool {
        matches!(self, PlaneMask::Stencil | PlaneMask::Both)
    }
}

/// Determine whether a UAV description targets the stencil plane of a two-plane depth resource.
pub(super) fn is_stencil_uav_format(fmt: u32, plane_slice: u32) -> bool {
    plane_slice == 1 || matches!(fmt, 22 | 47 | 60..=64 | 120)
}

/// Map a requested console/PC UAV format to a legal typed format on the chosen color twin.
pub(super) fn twin_uav_format(twin_typeless_fmt: i32, requested_fmt: u32) -> i32 {
    match twin_typeless_fmt {
        // R8_TYPELESS stencil twin
        60 => match requested_fmt {
            61 => DXGI_FORMAT_R8_UNORM.0,
            63 => DXGI_FORMAT_R8_SNORM.0,
            64 => DXGI_FORMAT_R8_SINT.0,
            _ => DXGI_FORMAT_R8_UINT.0,
        },
        // R16_TYPELESS depth twin
        53 => match requested_fmt {
            54 => DXGI_FORMAT_R16_FLOAT.0,
            57 => DXGI_FORMAT_R16_UINT.0,
            59 => DXGI_FORMAT_R16_SINT.0,
            _ => DXGI_FORMAT_R16_UNORM.0,
        },
        // R32_TYPELESS depth twin
        39 => match requested_fmt {
            42 => DXGI_FORMAT_R32_UINT.0,
            43 => DXGI_FORMAT_R32_SINT.0,
            _ => DXGI_FORMAT_R32_FLOAT.0,
        },
        other => other,
    }
}

/// Classify which plane(s) a D3D12 subresource index belongs to.
pub(super) fn plane_for_subresource(sub: u32, mips: u32, array_size: u32, two_plane: bool) -> PlaneMask {
    if !two_plane {
        return PlaneMask::Depth;
    }
    if sub == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES {
        return PlaneMask::Both;
    }
    let per_plane = mips.max(1).saturating_mul(array_size.max(1));
    if sub >= per_plane {
        PlaneMask::Stencil
    } else {
        PlaneMask::Depth
    }
}

struct Entry {
    device: usize,
    plan: TwinPlan,
    width: u64,
    height: u32,
    array_size: u16,
    mip_levels: u16,
    depth_twin: Option<ID3D12Resource>,
    stencil_twin: Option<ID3D12Resource>,
    /// Last known PC resource state per subresource for plane 0 (depth) and plane 1 (stencil).
    pc_states_depth: Vec<u32>,
    pc_states_stencil: Vec<u32>,
    /// True when the color twin may contain newer UAV writes than the depth-stencil resource.
    depth_twin_dirty: bool,
    stencil_twin_dirty: bool,
    /// True when the depth-stencil resource may contain newer DSV/clear writes than the twin.
    depth_plane_dirty: bool,
    stencil_plane_dirty: bool,
    /// True per subresource if a split barrier (`BEGIN_ONLY`) involving `UNORDERED_ACCESS` was suppressed.
    split_uav_depth: Vec<bool>,
    split_uav_stencil: Vec<bool>,
    /// Lazy DSV/SRV/UAV descriptor heaps for `FullscreenPass` sync on `R24G8_TYPELESS`.
    pass_dsv_heap: Option<ID3D12DescriptorHeap>,
    pass_srv_uav_heap: Option<ID3D12DescriptorHeap>,
    /// LOCAL agent I: snapshots of the depth / stencil twin taken after a chosen compute program
    /// (`XCL_LOCAL_TWIN_SNAP_CS`), dumped with the targets
    local_snap: [Option<ID3D12Resource>; 2],
}

// D3D12 resources and descriptor heaps are free-threaded.
unsafe impl Send for Entry {}

impl Entry {
    fn new(device: usize, plan: TwinPlan, width: u64, height: u32, array_size: u16, mip_levels: u16, initial_pc_state: u32, started_as_uav: bool) -> Self {
        let array_size = array_size.max(1);
        let mip_levels = mip_levels.max(1);
        let per_plane = (mip_levels as usize).saturating_mul(array_size as usize).max(1);
        let stencil_len = if plan.two_plane { per_plane } else { 0 };
        Self {
            device,
            plan,
            width,
            height,
            array_size,
            mip_levels,
            depth_twin: None,
            stencil_twin: None,
            pc_states_depth: vec![initial_pc_state; per_plane],
            pc_states_stencil: vec![initial_pc_state; stencil_len],
            depth_twin_dirty: started_as_uav,
            stencil_twin_dirty: started_as_uav && plan.two_plane,
            depth_plane_dirty: !started_as_uav && (initial_pc_state & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0),
            stencil_plane_dirty: !started_as_uav && plan.two_plane && (initial_pc_state & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0),
            split_uav_depth: vec![false; per_plane],
            split_uav_stencil: vec![false; stencil_len],
            pass_dsv_heap: None,
            pass_srv_uav_heap: None,
            local_snap: [None, None], // LOCAL agent I
        }
    }

    fn subresources_per_plane(&self) -> u32 {
        (self.mip_levels as u32).saturating_mul(self.array_size as u32).max(1)
    }

    fn local_subs_for(&self, subresource: u32, plane: u32) -> std::ops::Range<usize> {
        let per_plane = self.subresources_per_plane();
        if subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES {
            return 0..per_plane as usize;
        }
        let base = plane * per_plane;
        if subresource >= base && subresource < base + per_plane {
            let idx = (subresource - base) as usize;
            idx..idx + 1
        } else {
            0..0
        }
    }
}

/// Action queued by `filter_barrier` to run on the command list.
/// Each `(local_sub, before_state, after_state)` entry names the exact plane-local subresource
/// index and its tracked PC `StateBefore` / `StateAfter` so partial subresource barriers and
/// per-plane states never corrupt sibling subresources or planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SyncOp {
    TwinToDepth {
        res: usize,
        depth: Vec<(u32, u32, u32)>,
        stencil: Vec<(u32, u32, u32)>,
    },
    DepthToTwin {
        res: usize,
        depth: Vec<(u32, u32, u32)>,
        stencil: Vec<(u32, u32, u32)>,
    },
    TransitionOnly {
        res: usize,
        depth: Vec<(u32, u32, u32)>,
        stencil: Vec<(u32, u32, u32)>,
    },
}

static ENTRIES: Mutex<Option<HashMap<usize, Entry>>> = Mutex::new(None);
static COUNT: AtomicUsize = AtomicUsize::new(0);
/// DSV CPU handle -> resource raw pointer.
static DSV_TO_RES: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// Whether Depth+UAV color twins are enabled (`XCL_DEPTH_UAV_TWIN=0` turns them off).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_DEPTH_UAV_TWIN").is_ok_and(|v| v.trim() == "0"))
}

pub fn any() -> bool {
    COUNT.load(Ordering::Relaxed) != 0
}

/// Register a newly created console depth+UAV texture (`flags & 6 == 6`, single-sample, 2D).
pub(super) fn register_if_depth_uav(device: *mut c_void, res: *mut c_void, console_desc: &[u8; 0x38], initial_pc_state: u32, console_initial_state: u32) {
    if !enabled() || device.is_null() || res.is_null() {
        return;
    }
    let dim = u32::from_le_bytes(console_desc[0..4].try_into().unwrap());
    let flags = u32::from_le_bytes(console_desc[0x30..0x34].try_into().unwrap());
    let samples = u32::from_le_bytes(console_desc[0x24..0x28].try_into().unwrap());
    if dim != D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32 || (flags & (FLAG_DEPTH_STENCIL | FLAG_UAV)) != (FLAG_DEPTH_STENCIL | FLAG_UAV) || samples > 1 {
        return;
    }
    let console_fmt = u32::from_le_bytes(console_desc[0x20..0x24].try_into().unwrap());
    let Some(plan) = plan_for_console_format(console_fmt) else {
        return;
    };
    let width = u64::from_le_bytes(console_desc[0x10..0x18].try_into().unwrap());
    let height = u32::from_le_bytes(console_desc[0x18..0x1C].try_into().unwrap());
    let array_size = u16::from_le_bytes(console_desc[0x1C..0x1E].try_into().unwrap()).max(1);
    let mip_levels = u16::from_le_bytes(console_desc[0x1E..0x20].try_into().unwrap()).max(1);
    let started_as_uav = console_initial_state & STATE_UAV != 0;

    let mut g = ENTRIES.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    let prev = map.insert(
        res as usize,
        Entry::new(device as usize, plan, width, height, array_size, mip_levels, initial_pc_state, started_as_uav),
    );
    if prev.is_none() {
        COUNT.fetch_add(1, Ordering::Relaxed);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    let n = LOGGED.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_power_of_two() || (width >= 1920 && local_on()) { // LOCAL agent I: every screen-sized one
        crate::xlog_info!(
            "D3D12.X: tracked depth+UAV texture {res:p} ({width}x{height}x{array_size}, {mip_levels} mips, console fmt {console_fmt}, two_plane {}) for color-twin UAV sync ({} tracked)",
            plan.two_plane,
            map.len()
        );
    }
}

/// Remove tracking when a resource is released for good (twins go through `res_release` -> `late_release`).
pub(super) fn forget(res: usize) {
    if !any() {
        return;
    }
    let removed = {
        let mut g = ENTRIES.lock().unwrap();
        let r = g.as_mut().and_then(|m| m.remove(&res));
        if r.is_some() {
            COUNT.fetch_sub(1, Ordering::Relaxed);
        }
        r
    };
    if let Some(mut e) = removed {
        if let Some(m) = DSV_TO_RES.lock().unwrap().as_mut() {
            m.retain(|_, &mut v| v != res);
        }
        // Defer release of internal unadopted pass descriptor heaps until in-flight GPU work completes.
        unsafe {
            if let Some(h) = e.pass_dsv_heap.take() {
                super::late_release::hold(h.into_raw() as usize);
            }
            if let Some(h) = e.pass_srv_uav_heap.take() {
                super::late_release::hold(h.into_raw() as usize);
            }
        }
        drop(e);
    }
}

/// Record DSV handle -> resource for `OMSetRenderTargets` and `ClearDepthStencilView` hooks.
pub(super) fn note_dsv(dsv_cpu: usize, res: usize) {
    if !any() || dsv_cpu == 0 {
        return;
    }
    let tracked = ENTRIES.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&res));
    let mut g = DSV_TO_RES.lock().unwrap();
    if tracked {
        g.get_or_insert_with(HashMap::new).insert(dsv_cpu, res);
    } else if let Some(m) = g.as_mut() {
        m.remove(&dsv_cpu);
    }
}

unsafe fn create_twin_resource(device: *mut c_void, width: u64, height: u32, array_size: u16, mip_levels: u16, format: i32) -> Option<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width,
        Height: height,
        DepthOrArraySize: array_size,
        MipLevels: mip_levels,
        Format: DXGI_FORMAT(format),
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut out: *mut c_void = std::ptr::null_mut();
    let hr = real::dev_create_committed_resource(
        device,
        &heap as *const _ as usize,
        0,
        &desc as *const _ as usize,
        STATE_UAV as usize,
        0,
        &ID3D12Resource::IID as *const _ as *const windows_sys::core::GUID,
        &mut out,
    );
    if hr < 0 || out.is_null() {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            crate::xlog_warn!("D3D12.X: depth+UAV twin allocation ({width}x{height}x{array_size}, fmt {format}) failed ({hr:#x})");
        }
        return None;
    }
    Some(ID3D12Resource::from_raw(out))
}

/// Route a `CreateUnorderedAccessView` on a tracked depth+UAV resource to its depth or stencil
/// color twin, returning `(twin_raw_ptr, patched_uav_desc)` when handled.
pub(super) unsafe fn resolve_uav(
    dev: *mut c_void,
    res: *mut c_void,
    orig_desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<(*mut c_void, D3D12_UNORDERED_ACCESS_VIEW_DESC)> {
    if !any() || res.is_null() {
        return None;
    }
    let key = res as usize;
    let plane_slice = orig_desc.map_or(0, |d| match d.ViewDimension {
        D3D12_UAV_DIMENSION_TEXTURE2D => d.Anonymous.Texture2D.PlaneSlice,
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => d.Anonymous.Texture2DArray.PlaneSlice,
        _ => 0,
    });
    let req_fmt = orig_desc.map_or(0, |d| d.Format.0 as u32);

    // Inspect plan and check whether the twin already exists before allocating outside the lock.
    let (need_alloc, is_stencil, twin_fmt, width, height, array_size, mip_levels, device, existing_raw) = {
        let g = ENTRIES.lock().unwrap();
        let e = g.as_ref()?.get(&key)?;
        let stencil = e.plan.two_plane && is_stencil_uav_format(req_fmt, plane_slice);
        let twin_fmt = if stencil { e.plan.stencil_twin_fmt? } else { e.plan.depth_twin_fmt };
        let existing = if stencil {
            e.stencil_twin.as_ref().map(|r| r.as_raw())
        } else {
            e.depth_twin.as_ref().map(|r| r.as_raw())
        };
        (existing.is_none(), stencil, twin_fmt, e.width, e.height, e.array_size, e.mip_levels, if dev.is_null() { e.device as *mut c_void } else { dev }, existing)
    };

    let twin_raw = if let Some(raw) = existing_raw {
        raw
    } else if need_alloc {
        let created = create_twin_resource(device, width, height, array_size, mip_levels, twin_fmt)?;
        let mut g = ENTRIES.lock().unwrap();
        let Some(e) = g.as_mut().and_then(|m| m.get_mut(&key)) else {
            return None;
        };
        let slot = if is_stencil { &mut e.stencil_twin } else { &mut e.depth_twin };
        if slot.is_none() {
            *slot = Some(created);
        }
        slot.as_ref()?.as_raw()
    } else {
        return None;
    };

    let patched_fmt = twin_uav_format(twin_fmt, req_fmt);
    let mut patched = match orig_desc {
        Some(d) => *d,
        None => {
            let mut d = D3D12_UNORDERED_ACCESS_VIEW_DESC::default();
            if array_size > 1 {
                d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                    MipSlice: 0,
                    FirstArraySlice: 0,
                    ArraySize: array_size as u32,
                    PlaneSlice: 0,
                };
            } else {
                d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                d.Anonymous.Texture2D = D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 };
            }
            d
        }
    };
    patched.Format = DXGI_FORMAT(patched_fmt);
    match patched.ViewDimension {
        D3D12_UAV_DIMENSION_TEXTURE2D => patched.Anonymous.Texture2D.PlaneSlice = 0,
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => patched.Anonymous.Texture2DArray.PlaneSlice = 0,
        _ => {}
    }

    static ROUTED: AtomicU64 = AtomicU64::new(0);
    static ROUTED_BIG: AtomicU64 = AtomicU64::new(0); // LOCAL agent I
    let n = ROUTED.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_power_of_two() || (width >= 1920 && local_on() && ROUTED_BIG.fetch_add(1, Ordering::Relaxed) < 64) {
        if width >= 1920 && local_on() {
            // LOCAL agent I: the view the title asked for (format, dimension, plane) and what the twin got
            crate::xlog_info!(
                "LOCAL twin-view: {res:p} {width}x{height}: title view fmt {req_fmt} dim {} plane {plane_slice} -> {} twin {twin_raw:p} typeless fmt {twin_fmt}, view fmt {patched_fmt} (gfx10 class from it)",
                orig_desc.map_or(-1, |d| d.ViewDimension.0),
                if is_stencil { "stencil" } else { "depth" }
            );
        }
        crate::xlog_info!(
            "D3D12.X: routed {} UAV of depth+UAV texture {res:p} (req fmt {req_fmt} -> twin fmt {patched_fmt}) to color twin {twin_raw:p} ({} routed)",
            if is_stencil { "stencil" } else { "depth" },
            n + 1
        );
    }
    Some((twin_raw, patched))
}

/// Pure state-machine transition decision for a single barrier on a tracked depth+UAV texture.
/// Returns `(pass_through_barrier, needs_global_uav_barrier, optional_sync_op)`.
fn decide_transition(
    e: &mut Entry,
    subresource: u32,
    before: u32,
    after: u32,
    flags: u32,
    is_direct: bool,
) -> (bool, bool, Option<SyncOp>) {
    const BARRIER_BEGIN_ONLY: u32 = 1;
    const BARRIER_END_ONLY: u32 = 2;

    let planes = plane_for_subresource(subresource, e.mip_levels as u32, e.array_size as u32, e.plan.two_plane);
    let d_range = if planes.includes_depth() { e.local_subs_for(subresource, 0) } else { 0..0 };
    let s_range = if e.plan.two_plane && planes.includes_stencil() { e.local_subs_for(subresource, 1) } else { 0..0 };
    let touches_uav = (before | after) & STATE_UAV != 0;

    // Non-DIRECT lists (COMPUTE / COPY / BUNDLE) cannot transition a depth-stencil resource
    // to/from DEPTH_* or UNORDERED_ACCESS; suppress invalid barriers and defer sync to DIRECT.
    if !is_direct {
        if touches_uav {
            if after & STATE_UAV != 0 {
                if !d_range.is_empty() && e.depth_twin.is_some() {
                    e.depth_twin_dirty = true;
                }
                if !s_range.is_empty() && e.stencil_twin.is_some() {
                    e.stencil_twin_dirty = true;
                }
            }
            return (false, true, None);
        }
        return (false, false, None);
    }

    // Handle split barriers touching UNORDERED_ACCESS: suppress BEGIN_ONLY and perform the
    // complete atomic transition + sync at END_ONLY.
    if flags == BARRIER_BEGIN_ONLY {
        if touches_uav {
            for s in d_range {
                e.split_uav_depth[s] = true;
            }
            for s in s_range {
                e.split_uav_stencil[s] = true;
            }
            return (false, false, None);
        }
        return (true, false, None);
    }
    if flags == BARRIER_END_ONLY {
        let was_split_uav = d_range.clone().any(|s| e.split_uav_depth[s]) || s_range.clone().any(|s| e.split_uav_stencil[s]);
        if !touches_uav && !was_split_uav {
            for s in d_range.clone() {
                e.pc_states_depth[s] = after;
            }
            if !d_range.is_empty() && after & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0 {
                e.depth_plane_dirty = true;
            }
            for s in s_range.clone() {
                e.pc_states_stencil[s] = after;
            }
            if !s_range.is_empty() && after & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0 {
                e.stencil_plane_dirty = true;
            }
            return (true, false, None);
        }
        for s in d_range.clone() {
            e.split_uav_depth[s] = false;
        }
        for s in s_range.clone() {
            e.split_uav_stencil[s] = false;
        }
    }

    let collect_subs = |states: &mut [u32], range: std::ops::Range<usize>, target: u32| -> Vec<(u32, u32, u32)> {
        let mut v = Vec::with_capacity(range.len());
        for s in range {
            let prev = states[s];
            states[s] = target;
            v.push((s as u32, prev, target));
        }
        v
    };

    let leaving_uav = before & STATE_UAV != 0 && after & STATE_UAV == 0;
    let entering_uav = before & STATE_UAV == 0 && after & STATE_UAV != 0;

    if leaving_uav {
        let need_d = !d_range.is_empty() && e.depth_twin.is_some();
        let need_s = !s_range.is_empty() && e.stencil_twin.is_some();
        let depth = collect_subs(&mut e.pc_states_depth, d_range.clone(), after);
        let stencil = collect_subs(&mut e.pc_states_stencil, s_range.clone(), after);
        if !d_range.is_empty() {
            e.depth_twin_dirty = false;
            if after & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0 {
                e.depth_plane_dirty = true;
            }
        }
        if !s_range.is_empty() {
            e.stencil_twin_dirty = false;
            if after & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0 {
                e.stencil_plane_dirty = true;
            }
        }
        let op = if need_d || need_s {
            Some(SyncOp::TwinToDepth { res: 0, depth, stencil })
        } else {
            Some(SyncOp::TransitionOnly { res: 0, depth, stencil })
        };
        return (false, true, op);
    }

    if entering_uav {
        // Transition depth resource from each subresource's tracked PC state (`prev`) to
        // `DEPTH_WRITE` as its resting state while UAV is active.
        let need_d = !d_range.is_empty() && e.depth_twin.is_some() && (e.depth_plane_dirty || before & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0);
        let need_s = !s_range.is_empty() && e.stencil_twin.is_some() && (e.stencil_plane_dirty || before & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0);
        let depth = collect_subs(&mut e.pc_states_depth, d_range.clone(), STATE_DEPTH_WRITE);
        let stencil = collect_subs(&mut e.pc_states_stencil, s_range.clone(), STATE_DEPTH_WRITE);
        if !d_range.is_empty() {
            e.depth_twin_dirty = e.depth_twin.is_some();
            e.depth_plane_dirty = false;
        }
        if !s_range.is_empty() {
            e.stencil_twin_dirty = e.stencil_twin.is_some();
            e.stencil_plane_dirty = false;
        }
        let op = if need_d || need_s {
            Some(SyncOp::DepthToTwin { res: 0, depth, stencil })
        } else {
            Some(SyncOp::TransitionOnly { res: 0, depth, stencil })
        };
        return (false, true, op);
    }

    // Neither before nor after is UNORDERED_ACCESS. If a twin became dirty (e.g. on an async
    // compute list), flush it into the depth resource now as part of this transition.
    let flush_d = !d_range.is_empty() && e.depth_twin_dirty && e.depth_twin.is_some();
    let flush_s = !s_range.is_empty() && e.stencil_twin_dirty && e.stencil_twin.is_some();
    let depth = collect_subs(&mut e.pc_states_depth, d_range.clone(), after);
    let stencil = collect_subs(&mut e.pc_states_stencil, s_range.clone(), after);
    if !d_range.is_empty() {
        if flush_d {
            e.depth_twin_dirty = false;
        }
        if after & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0 {
            e.depth_plane_dirty = true;
        }
    }
    if !s_range.is_empty() {
        if flush_s {
            e.stencil_twin_dirty = false;
        }
        if after & (STATE_DEPTH_WRITE | STATE_COPY_DEST) != 0 {
            e.stencil_plane_dirty = true;
        }
    }
    if flush_d || flush_s {
        let op = SyncOp::TwinToDepth { res: 0, depth, stencil };
        return (false, true, Some(op));
    }

    (true, false, None)
}

/// Inspect a PC transition barrier `[ty, flags, res_lo, res_hi, sub, before, after, _]` inside
/// `list_x::resource_barrier_x`. Returns `(pass_through, emit_global_uav_barrier)` and pushes
/// any required sync operation onto `ops`.
pub(super) fn filter_transition(
    b: &[u32; 8],
    is_direct: bool,
    ops: &mut Vec<SyncOp>,
) -> (bool, bool) {
    if !any() {
        return (true, false);
    }
    let res = (b[2] as u64 | (b[3] as u64) << 32) as usize;
    let mut g = ENTRIES.lock().unwrap();
    let Some(e) = g.as_mut().and_then(|m| m.get_mut(&res)) else {
        return (true, false);
    };
    let (pass, uav, op) = decide_transition(e, b[4], b[5], b[6], b[1], is_direct);
    if let Some(mut op) = op {
        match &mut op {
            SyncOp::TwinToDepth { res: r, .. } | SyncOp::DepthToTwin { res: r, .. } | SyncOp::TransitionOnly { res: r, .. } => *r = res,
        }
        ops.push(op);
    }
    (pass, uav)
}

/// Route a resource-specific UAV barrier targeting a tracked Depth+UAV resource to both its
/// materialized depth and stencil color twins, and mark those twins dirty so subsequent DSV/SRV
/// transitions flush their latest UAV writes back into the depth-stencil resource.
pub(super) fn on_uav_barrier(res: usize) -> [usize; 2] {
    if !any() || res == 0 {
        return [0, 0];
    }
    let mut g = ENTRIES.lock().unwrap();
    let Some(e) = g.as_mut().and_then(|m| m.get_mut(&res)) else {
        return [0, 0];
    };
    let d = match &e.depth_twin {
        Some(t) => {
            e.depth_twin_dirty = true;
            t.as_raw() as usize
        }
        None => 0,
    };
    let s = match &e.stencil_twin {
        Some(t) => {
            e.stencil_twin_dirty = true;
            t.as_raw() as usize
        }
        None => 0,
    };
    [d, s]
}

/// Before `OMSetRenderTargets` binds `dsv_cpu`, flush any dirty color twin into the depth resource.
pub(super) unsafe fn before_bind_dsv(list: *mut c_void, dsv_cpu: usize) {
    if !any() || dsv_cpu == 0 {
        return;
    }
    let Some(res) = DSV_TO_RES.lock().unwrap().as_ref().and_then(|m| m.get(&dsv_cpu).copied()) else {
        return;
    };
    // LOCAL agent I (`XCL_LOCAL_TWIN_AT_USE=1`): copy at use, see `local_sync_at_use`
    if local_at_use() {
        local_sync_at_use(list, res, LocalUse::DsvBind);
        return;
    }
    let op = {
        let mut g = ENTRIES.lock().unwrap();
        let Some(e) = g.as_mut().and_then(|m| m.get_mut(&res)) else {
            return;
        };
        // LOCAL agent G (`XCL_LOCAL_TWIN_FLUSH_D24=1`): the title's enhanced barriers never mark a
        // twin dirty here (only resource-specific UAV barriers / legacy transitions do), so for
        // D24S8 (FullscreenPass, R32 twin) the depth twin is flushed at every DSV bind; stencil not.
        let force = std::env::var_os("XCL_LOCAL_TWIN_FLUSH_D24").is_some()
            && e.plan.two_plane
            && matches!(e.plan.depth_sync, DepthSyncKind::FullscreenPass { .. })
            && e.depth_twin.is_some();
        if force {
            static NF: AtomicU64 = AtomicU64::new(0);
            let n = NF.fetch_add(1, Ordering::Relaxed);
            if n < 8 || n.is_power_of_two() {
                crate::xlog_info!("LOCAL twin-flush-d24: DSV bind of {res:#x} ({}x{}): depth twin flushed (dirty {} / stencil dirty {}), #{}", e.width, e.height, e.depth_twin_dirty, e.stencil_twin_dirty, n + 1);
            }
        }
        let d = (e.depth_twin_dirty || force) && e.depth_twin.is_some();
        let s = e.stencil_twin_dirty && e.stencil_twin.is_some() && !force;
        if !d && !s {
            return;
        }
        if d {
            e.depth_twin_dirty = false;
        }
        if s {
            e.stencil_twin_dirty = false;
        }
        let depth = if d {
            e.pc_states_depth.iter().enumerate().map(|(s, &st)| (s as u32, st, st)).collect()
        } else {
            Vec::new()
        };
        let stencil = if s {
            e.pc_states_stencil.iter().enumerate().map(|(s, &st)| (s as u32, st, st)).collect()
        } else {
            Vec::new()
        };
        SyncOp::TwinToDepth { res, depth, stencil }
    };
    execute_syncs(list, &[op]);
}

/// Record a `ClearDepthStencilView` on a tracked depth+UAV resource so cleared planes are marked
/// depth-dirty (and no longer wait for a stale twin flush).
pub(super) fn on_clear_dsv(dsv_cpu: usize, clear_flags: u32) {
    if !any() || dsv_cpu == 0 {
        return;
    }
    let Some(res) = DSV_TO_RES.lock().unwrap().as_ref().and_then(|m| m.get(&dsv_cpu).copied()) else {
        return;
    };
    let mut g = ENTRIES.lock().unwrap();
    if let Some(e) = g.as_mut().and_then(|m| m.get_mut(&res)) {
        if clear_flags & 1 != 0 {
            e.depth_twin_dirty = false;
            e.depth_plane_dirty = true;
        }
        if clear_flags & 2 != 0 && e.plan.two_plane {
            e.stencil_twin_dirty = false;
            e.stencil_plane_dirty = true;
        }
    }
}

/// RAII guard for `D3D12_TEXTURE_COPY_LOCATION` so `ManuallyDrop<Option<ID3D12Resource>>` is
/// always dropped cleanly without leaking COM references.
struct OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION);

impl Drop for OwnedCopyLoc {
    fn drop(&mut self) {
        unsafe { ManuallyDrop::drop(&mut self.0.pResource) };
    }
}

/// Transition specific plane-local subresources on `list` from their tracked `before` state to
/// `target_override` (if `Some`) or their tracked `after` state (if `None`).
unsafe fn transition_plane_subs(
    list: *mut c_void,
    res: usize,
    two_plane: bool,
    per_plane: u32,
    plane: u32,
    subs: &[(u32, u32, u32)],
    target_override: Option<u32>,
) {
    if subs.is_empty() {
        return;
    }
    if !two_plane && subs.len() == per_plane as usize {
        let first_b = subs[0].1;
        let first_a = target_override.unwrap_or(subs[0].2);
        if subs.iter().all(|&(_, b, a)| b == first_b && target_override.unwrap_or(a) == first_a) {
            if first_b != first_a {
                list_x::transition(list, res, D3D12_RESOURCE_STATES(first_b as i32), D3D12_RESOURCE_STATES(first_a as i32));
            }
            return;
        }
    }
    let base = plane * per_plane;
    let mut barriers = Vec::with_capacity(subs.len());
    for &(s, before, after) in subs {
        let dst = target_override.unwrap_or(after);
        if before == dst {
            continue;
        }
        barriers.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(&res),
                    Subresource: base + s,
                    StateBefore: D3D12_RESOURCE_STATES(before as i32),
                    StateAfter: D3D12_RESOURCE_STATES(dst as i32),
                }),
            },
        });
    }
    if !barriers.is_empty() {
        list_x::pc_resource_barriers(list, &barriers);
    }
}

/// Transition specific plane-local subresources from a common intermediate `from_state` back to
/// each subresource's tracked `after` state.
unsafe fn transition_plane_from(
    list: *mut c_void,
    res: usize,
    two_plane: bool,
    per_plane: u32,
    plane: u32,
    from_state: u32,
    subs: &[(u32, u32, u32)],
) {
    if subs.is_empty() {
        return;
    }
    if !two_plane && subs.len() == per_plane as usize {
        let first_a = subs[0].2;
        if subs.iter().all(|&(_, _, a)| a == first_a) {
            if from_state != first_a {
                list_x::transition(list, res, D3D12_RESOURCE_STATES(from_state as i32), D3D12_RESOURCE_STATES(first_a as i32));
            }
            return;
        }
    }
    let base = plane * per_plane;
    let mut barriers = Vec::with_capacity(subs.len());
    for &(s, _, after) in subs {
        if from_state == after {
            continue;
        }
        barriers.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(&res),
                    Subresource: base + s,
                    StateBefore: D3D12_RESOURCE_STATES(from_state as i32),
                    StateAfter: D3D12_RESOURCE_STATES(after as i32),
                }),
            },
        });
    }
    if !barriers.is_empty() {
        list_x::pc_resource_barriers(list, &barriers);
    }
}

/// Copy specific subresources (`subs` local indices) directly between two format-compatible
/// single-plane resources via `CopyTextureRegion` (or `CopyResource` if all subresources match).
unsafe fn copy_direct_subs(
    list: *mut c_void,
    dst_res: &ID3D12Resource,
    src_res: &ID3D12Resource,
    per_plane: u32,
    subs: &[(u32, u32, u32)],
) {
    if subs.len() == per_plane as usize {
        list_x::pc_copy_resource(list, dst_res.as_raw() as usize, src_res.as_raw() as usize);
        return;
    }
    for &(s, _, _) in subs {
        let dst_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(dst_res.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: s },
        });
        let src_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(src_res.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: s },
        });
        list_x::pc_copy_texture_region(list, &dst_loc.0, 0, 0, 0, &src_loc.0, std::ptr::null());
    }
}

/// Copy specific subresources (`subs`) on one plane (`plane`: 0 or 1) between `depth_res` and
/// `twin_res` via a staging buffer using `CopyTextureRegion`.
unsafe fn copy_plane_through_buffer(
    list: *mut c_void,
    dev: &ID3D12Device,
    dst_res: &ID3D12Resource,
    dst_plane: u32,
    src_res: &ID3D12Resource,
    src_plane: u32,
    per_plane: u32,
    subs: &[(u32, u32, u32)],
) -> bool {
    let dst_desc = dst_res.GetDesc();
    let src_desc = src_res.GetDesc();
    for &(s, _, _) in subs {
        let src_sub = src_plane * per_plane + s;
        let dst_sub = dst_plane * per_plane + s;
        let mut src_fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut dst_fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut src_rows = 0u32;
        let mut dst_rows = 0u32;
        let mut src_total = 0u64;
        let mut dst_total = 0u64;
        dev.GetCopyableFootprints(&src_desc, src_sub, 1, 0, Some(&mut src_fp), Some(&mut src_rows), None, Some(&mut src_total));
        dev.GetCopyableFootprints(&dst_desc, dst_sub, 1, 0, Some(&mut dst_fp), Some(&mut dst_rows), None, Some(&mut dst_total));
        if src_total == 0
            || src_total == u64::MAX
            || dst_total == 0
            || dst_total == u64::MAX
            || src_rows != dst_rows
            || src_fp.Footprint.RowPitch != dst_fp.Footprint.RowPitch
            || src_fp.Footprint.Width != dst_fp.Footprint.Width
            || src_fp.Footprint.Height != dst_fp.Footprint.Height
        {
            return false;
        }
        let needed = src_total.max(dst_total);
        let Some(scratch_raw) = list_x::acquire_scratch_for_copy(list, needed) else {
            return false;
        };
        let scratch_ptr = scratch_raw as *mut c_void;
        let Some(scratch) = ID3D12Resource::from_raw_borrowed(&scratch_ptr) else {
            return false;
        };
        list_x::transition(list, scratch_raw, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        {
            let src_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(src_res.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: src_sub },
            });
            let buf_dst_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(scratch.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: src_fp },
            });
            list_x::pc_copy_texture_region(list, &buf_dst_loc.0, 0, 0, 0, &src_loc.0, std::ptr::null());
        }

        list_x::transition(list, scratch_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        {
            let buf_src_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(scratch.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: dst_fp },
            });
            let dst_loc = OwnedCopyLoc(D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(dst_res.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: dst_sub },
            });
            list_x::pc_copy_texture_region(list, &dst_loc.0, 0, 0, 0, &buf_src_loc.0, std::ptr::null());
        }

        list_x::transition(list, scratch_raw, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    true
}

// ---------------------------------------------------------------------------------------
// Internal fullscreen depth-write & compute depth-read pipelines for R24G8_TYPELESS
// (console D16_UNORM_S8_UINT / D24_UNORM_S8_UINT plane 0)
// ---------------------------------------------------------------------------------------

struct SyncPipelines {
    gfx_rs: ID3D12RootSignature,
    gfx_pso: ID3D12PipelineState,
    cs_rs: ID3D12RootSignature,
    cs_pso: ID3D12PipelineState,
}

unsafe impl Send for SyncPipelines {}

static PIPELINES: Mutex<Option<HashMap<usize, Option<SyncPipelines>>>> = Mutex::new(None);

fn compile_sm5(src: &str, target: &str) -> Option<Vec<u8>> {
    unsafe {
        let mut blob = None;
        let t = format!("{target}\0");
        let r = D3DCompile(
            src.as_ptr() as *const c_void,
            src.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(b"main\0".as_ptr()),
            PCSTR(t.as_ptr()),
            0,
            0,
            &mut blob,
            None,
        );
        match (r, blob) {
            (Ok(()), Some(b)) => Some(std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec()),
            _ => None,
        }
    }
}

unsafe fn build_sync_pipelines(dev: &ID3D12Device) -> Option<SyncPipelines> {
    let vs = compile_sm5(
        "float4 main(uint id : SV_VertexID) : SV_Position {\n\
            float2 p = float2((id == 2) ? 3.0 : -1.0, (id == 1) ? 3.0 : -1.0);\n\
            return float4(p, 0.0, 1.0);\n\
        }\n",
        "vs_5_0",
    )?;
    let ps = compile_sm5(
        "Texture2DArray<float> src : register(t0);\n\
        cbuffer C : register(b0) { uint slice; uint mip; };\n\
        float main(float4 pos : SV_Position) : SV_Depth {\n\
            return src.Load(int4(int2(pos.xy), int(slice), int(mip)));\n\
        }\n",
        "ps_5_0",
    )?;
    let cs = compile_sm5(
        "Texture2DArray<float> src : register(t0);\n\
        RWTexture2DArray<float> dst : register(u0);\n\
        cbuffer C : register(b0) { uint w; uint h; uint slices; uint mip; };\n\
        [numthreads(8, 8, 1)]\n\
        void main(uint3 id : SV_DispatchThreadID) {\n\
            if (id.x < w && id.y < h && id.z < slices) {\n\
                dst[id] = src.Load(int4(int3(id), int(mip)));\n\
            }\n\
        }\n",
        "cs_5_0",
    )?;

    // Graphics RS: [0] 2 root constants (b0), [1] 1 SRV table (t0)
    let srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let gfx_params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 2 },
            },
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &srv_range },
            },
        },
    ];
    let gfx_rs_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: gfx_params.len() as u32,
        pParameters: gfx_params.as_ptr(),
        ..Default::default()
    };
    let mut blob = None;
    D3D12SerializeRootSignature(&gfx_rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).ok()?;
    let blob = blob?;
    let dev_raw = dev.as_raw();
    type CreateRsFn = unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const windows_sys::core::GUID, *mut *mut c_void) -> i32;
    let create_rs: CreateRsFn = std::mem::transmute(real::orig_fn(dev_raw, 16));
    let mut gfx_rs_raw = std::ptr::null_mut();
    if create_rs(dev_raw, 0, blob.GetBufferPointer(), blob.GetBufferSize(), &ID3D12RootSignature::IID as *const _ as *const _, &mut gfx_rs_raw) < 0 || gfx_rs_raw.is_null() {
        return None;
    }
    let gfx_rs = ID3D12RootSignature::from_raw(gfx_rs_raw);

    // Compute RS: [0] 4 root constants (b0), [1] 1 SRV table (t0), [2] 1 UAV table (u0)
    let uav_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let cs_params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 },
            },
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &srv_range },
            },
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &uav_range },
            },
        },
    ];
    let cs_rs_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: cs_params.len() as u32,
        pParameters: cs_params.as_ptr(),
        ..Default::default()
    };
    let mut cs_blob = None;
    D3D12SerializeRootSignature(&cs_rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut cs_blob, None).ok()?;
    let cs_blob = cs_blob?;
    let mut cs_rs_raw = std::ptr::null_mut();
    if create_rs(dev_raw, 0, cs_blob.GetBufferPointer(), cs_blob.GetBufferSize(), &ID3D12RootSignature::IID as *const _ as *const _, &mut cs_rs_raw) < 0 || cs_rs_raw.is_null() {
        return None;
    }
    let cs_rs = ID3D12RootSignature::from_raw(cs_rs_raw);

    // Graphics PSO: write SV_Depth unconditionally, stencil disabled so plane 1 stays intact.
    let mut gfx_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(gfx_rs.clone())),
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as *const c_void, BytecodeLength: vs.len() },
        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as *const c_void, BytecodeLength: ps.len() },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: false.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            StencilEnable: false.into(),
            ..Default::default()
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 0,
        DSVFormat: DXGI_FORMAT_D24_UNORM_S8_UINT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    type CreateGfxPsoFn = unsafe extern "system" fn(*mut c_void, *const D3D12_GRAPHICS_PIPELINE_STATE_DESC, *const windows_sys::core::GUID, *mut *mut c_void) -> i32;
    let create_gfx: CreateGfxPsoFn = std::mem::transmute(real::orig_fn(dev_raw, 10));
    let mut gfx_pso_raw = std::ptr::null_mut();
    let hr_gfx = create_gfx(dev_raw, &gfx_desc, &ID3D12PipelineState::IID as *const _ as *const _, &mut gfx_pso_raw);
    ManuallyDrop::drop(&mut gfx_desc.pRootSignature);
    if hr_gfx < 0 || gfx_pso_raw.is_null() {
        return None;
    }
    let gfx_pso = ID3D12PipelineState::from_raw(gfx_pso_raw);

    let mut cs_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(cs_rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() },
        ..Default::default()
    };
    type CreateCsPsoFn = unsafe extern "system" fn(*mut c_void, *const D3D12_COMPUTE_PIPELINE_STATE_DESC, *const windows_sys::core::GUID, *mut *mut c_void) -> i32;
    let create_cs: CreateCsPsoFn = std::mem::transmute(real::orig_fn(dev_raw, 11));
    let mut cs_pso_raw = std::ptr::null_mut();
    let hr_cs = create_cs(dev_raw, &cs_desc, &ID3D12PipelineState::IID as *const _ as *const _, &mut cs_pso_raw);
    ManuallyDrop::drop(&mut cs_desc.pRootSignature);
    if hr_cs < 0 || cs_pso_raw.is_null() {
        return None;
    }
    let cs_pso = ID3D12PipelineState::from_raw(cs_pso_raw);

    Some(SyncPipelines { gfx_rs, gfx_pso, cs_rs, cs_pso })
}

/// Ensure per-resource internal unadopted PC DSV + SRV/UAV descriptor heaps exist for `R24G8_TYPELESS` sync.
unsafe fn ensure_pass_heaps(
    dev: &ID3D12Device,
    depth_res: &ID3D12Resource,
    twin_res: &ID3D12Resource,
    e: &mut Entry,
    twin_srv_fmt: i32,
    twin_uav_fmt: i32,
) -> Option<(ID3D12DescriptorHeap, ID3D12DescriptorHeap)> {
    let dev_raw = dev.as_raw();
    type CreateHeapFn = unsafe extern "system" fn(*mut c_void, *const D3D12_DESCRIPTOR_HEAP_DESC, *const windows_sys::core::GUID, *mut *mut c_void) -> i32;
    type IncSizeFn = unsafe extern "system" fn(*mut c_void, D3D12_DESCRIPTOR_HEAP_TYPE) -> u32;
    type CreateDsvFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_DEPTH_STENCIL_VIEW_DESC, usize);
    type CreateSrvFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, usize);
    type CreateUavFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const D3D12_UNORDERED_ACCESS_VIEW_DESC, usize);
    let create_heap: CreateHeapFn = std::mem::transmute(real::orig_fn(dev_raw, 14));
    let inc_size: IncSizeFn = std::mem::transmute(real::orig_fn(dev_raw, 15));
    let create_srv: CreateSrvFn = std::mem::transmute(real::orig_fn(dev_raw, 18));
    let create_uav: CreateUavFn = std::mem::transmute(real::orig_fn(dev_raw, 19));
    let create_dsv: CreateDsvFn = std::mem::transmute(real::orig_fn(dev_raw, 21));

    let mips = e.mip_levels as u32;
    let slices = e.array_size as u32;
    if e.pass_dsv_heap.is_none() {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            NumDescriptors: mips * slices,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let mut raw = std::ptr::null_mut();
        if create_heap(dev_raw, &desc, &ID3D12DescriptorHeap::IID as *const _ as *const _, &mut raw) < 0 || raw.is_null() {
            return None;
        }
        let dsv_heap = ID3D12DescriptorHeap::from_raw(raw);
        let inc = inc_size(dev_raw, D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
        let base = dsv_heap.GetCPUDescriptorHandleForHeapStart().ptr;
        for m in 0..mips {
            for a in 0..slices {
                let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D24_UNORM_S8_UINT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                    Flags: D3D12_DSV_FLAG_NONE,
                    Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_DSV { MipSlice: m, FirstArraySlice: a, ArraySize: 1 },
                    },
                };
                let idx = (m * slices + a) as usize;
                create_dsv(dev_raw, depth_res.as_raw(), &dsv, base + idx * inc);
            }
        }
        e.pass_dsv_heap = Some(dsv_heap);
    }
    if e.pass_srv_uav_heap.is_none() {
        // Slot 0: twin SRV, Slot 1: depth plane 0 SRV (R24_UNORM_X8_TYPELESS), Slots 2..2+mips: twin UAV per mip
        let total = 2 + mips;
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: total,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        let mut raw = std::ptr::null_mut();
        if create_heap(dev_raw, &desc, &ID3D12DescriptorHeap::IID as *const _ as *const _, &mut raw) < 0 || raw.is_null() {
            return None;
        }
        let heap = ID3D12DescriptorHeap::from_raw(raw);
        let inc = inc_size(dev_raw, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let base = heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let twin_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(twin_srv_fmt),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MostDetailedMip: 0, MipLevels: mips, FirstArraySlice: 0, ArraySize: slices, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
            },
        };
        create_srv(dev_raw, twin_res.as_raw(), &twin_srv, base);
        let depth_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MostDetailedMip: 0, MipLevels: mips, FirstArraySlice: 0, ArraySize: slices, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
            },
        };
        create_srv(dev_raw, depth_res.as_raw(), &depth_srv, base + inc);
        for m in 0..mips {
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT(twin_uav_fmt),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_UAV { MipSlice: m, FirstArraySlice: 0, ArraySize: slices, PlaneSlice: 0 },
                },
            };
            create_uav(dev_raw, twin_res.as_raw(), std::ptr::null_mut(), &uav, base + (2 + m as usize) * inc);
        }
        e.pass_srv_uav_heap = Some(heap);
    }
    Some((e.pass_dsv_heap.clone()?, e.pass_srv_uav_heap.clone()?))
}

unsafe fn get_pipelines(dev: &ID3D12Device) -> Option<(ID3D12RootSignature, ID3D12PipelineState, ID3D12RootSignature, ID3D12PipelineState)> {
    let key = dev.as_raw() as usize;
    let mut g = PIPELINES.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    let entry = map.entry(key).or_insert_with(|| build_sync_pipelines(dev));
    let p = entry.as_ref()?;
    Some((p.gfx_rs.clone(), p.gfx_pso.clone(), p.cs_rs.clone(), p.cs_pso.clone()))
}

/// Execute queued depth+UAV synchronization ops on a `DIRECT` command list.
pub(super) unsafe fn execute_syncs(list: *mut c_void, ops: &[SyncOp]) {
    if ops.is_empty() || list.is_null() {
        return;
    }
    let Some(dev) = list_x::device_of_list(list) else {
        return;
    };
    let dev_raw = dev.as_raw();
    type IncSizeFn = unsafe extern "system" fn(*mut c_void, D3D12_DESCRIPTOR_HEAP_TYPE) -> u32;
    let inc_size: IncSizeFn = std::mem::transmute(real::orig_fn(dev_raw, 15));
    let mut clobbered_gfx = false;
    let mut clobbered_cs = false;

    for op in ops {
        match op {
            SyncOp::TransitionOnly { res, depth, stencil } => {
                let res = *res;
                let Some((two_plane, per_plane)) = ({
                    let g = ENTRIES.lock().unwrap();
                    g.as_ref().and_then(|m| m.get(&res)).map(|e| (e.plan.two_plane, e.subresources_per_plane()))
                }) else {
                    continue;
                };
                if !depth.is_empty() {
                    transition_plane_subs(list, res, two_plane, per_plane, 0, depth, None);
                }
                if two_plane && !stencil.is_empty() {
                    transition_plane_subs(list, res, two_plane, per_plane, 1, stencil, None);
                }
            }
            SyncOp::TwinToDepth { res, depth, stencil } => {
                let res = *res;
                let res_ptr = res as *mut c_void;
                let Some(depth_res) = ID3D12Resource::from_raw_borrowed(&res_ptr) else {
                    continue;
                };
                let snapshot = {
                    let g = ENTRIES.lock().unwrap();
                    g.as_ref().and_then(|m| m.get(&res)).map(|e| {
                        (
                            e.plan,
                            e.width,
                            e.height,
                            e.array_size,
                            e.mip_levels,
                            e.subresources_per_plane(),
                            e.depth_twin.clone(),
                            e.stencil_twin.clone(),
                        )
                    })
                };
                let Some((plan, width, height, array_size, mip_levels, per_plane, depth_twin, stencil_twin)) = snapshot else {
                    continue;
                };

                if !depth.is_empty() {
                    if let Some(twin) = depth_twin {
                        match plan.depth_sync {
                            DepthSyncKind::DirectCopy => {
                                let twin_raw = twin.as_raw() as usize;
                                list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                                transition_plane_subs(list, res, false, per_plane, 0, depth, Some(STATE_COPY_DEST));
                                copy_direct_subs(list, depth_res, &twin, per_plane, depth);
                                list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                                transition_plane_from(list, res, false, per_plane, 0, STATE_COPY_DEST, depth);
                            }
                            DepthSyncKind::BufferThrough => {
                                let twin_raw = twin.as_raw() as usize;
                                list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                                transition_plane_subs(list, res, true, per_plane, 0, depth, Some(STATE_COPY_DEST));
                                let _ = copy_plane_through_buffer(list, &dev, depth_res, 0, &twin, 0, per_plane, depth);
                                list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                                transition_plane_from(list, res, true, per_plane, 0, STATE_COPY_DEST, depth);
                            }
                            DepthSyncKind::FullscreenPass { twin_srv_fmt, twin_uav_fmt } => {
                                let heaps = {
                                    let mut g = ENTRIES.lock().unwrap();
                                    g.as_mut().and_then(|m| m.get_mut(&res)).and_then(|e| ensure_pass_heaps(&dev, depth_res, &twin, e, twin_srv_fmt, twin_uav_fmt))
                                };
                                if let (Some((gfx_rs, gfx_pso, _, _)), Some((dsv_heap, srv_heap))) = (get_pipelines(&dev), heaps) {
                                    let twin_raw = twin.as_raw() as usize;
                                    list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATES(STATE_SRV_ALL as i32));
                                    transition_plane_subs(list, res, true, per_plane, 0, depth, Some(STATE_DEPTH_WRITE));
                                    let srv_raw = srv_heap.as_raw() as usize;
                                    let set_heaps: unsafe extern "system" fn(*mut c_void, u32, *const usize) = std::mem::transmute(real::orig_fn(list, 28));
                                    let set_gfx_rs: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list, 30));
                                    let set_pso: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list, 25));
                                    let set_topo: unsafe extern "system" fn(*mut c_void, u32) = std::mem::transmute(real::orig_fn(list, 20));
                                    let set_gfx_table: unsafe extern "system" fn(*mut c_void, u32, u64) = std::mem::transmute(real::orig_fn(list, 32));
                                    let set_vp: unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT) = std::mem::transmute(real::orig_fn(list, 21));
                                    let set_sc: unsafe extern "system" fn(*mut c_void, u32, *const RECT) = std::mem::transmute(real::orig_fn(list, 22));
                                    let set_rt: unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize) = std::mem::transmute(real::orig_fn(list, 46));
                                    let set_gfx_consts: unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32) = std::mem::transmute(real::orig_fn(list, 36));
                                    let draw: unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32) = std::mem::transmute(real::orig_fn(list, 12));

                                    set_heaps(list, 1, &srv_raw);
                                    set_gfx_rs(list, gfx_rs.as_raw());
                                    set_pso(list, gfx_pso.as_raw());
                                    set_topo(list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0 as u32);
                                    set_gfx_table(list, 1, srv_heap.GetGPUDescriptorHandleForHeapStart().ptr);
                                    let dsv_base = dsv_heap.GetCPUDescriptorHandleForHeapStart().ptr;
                                    let dsv_inc = inc_size(dev_raw, D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                                    let mips_u32 = mip_levels as u32;
                                    for &(local_sub, _, _) in depth {
                                        let m = local_sub % mips_u32;
                                        let a = local_sub / mips_u32;
                                        let mw = ((width as u32) >> m).max(1);
                                        let mh = (height >> m).max(1);
                                        let vp = D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: mw as f32, Height: mh as f32, MinDepth: 0.0, MaxDepth: 1.0 };
                                        let sc = RECT { left: 0, top: 0, right: mw as i32, bottom: mh as i32 };
                                        set_vp(list, 1, &vp);
                                        set_sc(list, 1, &sc);
                                        let dsv = dsv_base + ((m * array_size as u32 + a) as usize) * dsv_inc;
                                        set_rt(list, 0, std::ptr::null(), 0, &dsv);
                                        let c = [a, m];
                                        set_gfx_consts(list, 0, 2, c.as_ptr(), 0);
                                        draw(list, 3, 1, 0, 0);
                                    }
                                    list_x::transition(list, twin_raw, D3D12_RESOURCE_STATES(STATE_SRV_ALL as i32), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                                    transition_plane_from(list, res, true, per_plane, 0, STATE_DEPTH_WRITE, depth);
                                    clobbered_gfx = true;
                                } else {
                                    transition_plane_subs(list, res, true, per_plane, 0, depth, None);
                                }
                            }
                        }
                    } else {
                        transition_plane_subs(list, res, plan.two_plane, per_plane, 0, depth, None);
                    }
                }

                if plan.two_plane && !stencil.is_empty() {
                    if let Some(twin) = stencil_twin {
                        let twin_raw = twin.as_raw() as usize;
                        list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                        transition_plane_subs(list, res, true, per_plane, 1, stencil, Some(STATE_COPY_DEST));
                        let _ = copy_plane_through_buffer(list, &dev, depth_res, 1, &twin, 0, per_plane, stencil);
                        list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                        transition_plane_from(list, res, true, per_plane, 1, STATE_COPY_DEST, stencil);
                    } else {
                        transition_plane_subs(list, res, true, per_plane, 1, stencil, None);
                    }
                }
            }
            SyncOp::DepthToTwin { res, depth, stencil } => {
                let res = *res;
                let res_ptr = res as *mut c_void;
                let Some(depth_res) = ID3D12Resource::from_raw_borrowed(&res_ptr) else {
                    continue;
                };
                let snapshot = {
                    let g = ENTRIES.lock().unwrap();
                    g.as_ref().and_then(|m| m.get(&res)).map(|e| {
                        (
                            e.plan,
                            e.width,
                            e.height,
                            e.array_size,
                            e.mip_levels,
                            e.subresources_per_plane(),
                            e.depth_twin.clone(),
                            e.stencil_twin.clone(),
                        )
                    })
                };
                let Some((plan, width, height, array_size, mip_levels, per_plane, depth_twin, stencil_twin)) = snapshot else {
                    continue;
                };

                if !depth.is_empty() {
                    if let Some(twin) = depth_twin {
                        match plan.depth_sync {
                            DepthSyncKind::DirectCopy => {
                                let twin_raw = twin.as_raw() as usize;
                                transition_plane_subs(list, res, false, per_plane, 0, depth, Some(STATE_COPY_SOURCE));
                                list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
                                copy_direct_subs(list, &twin, depth_res, per_plane, depth);
                                list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                                transition_plane_from(list, res, false, per_plane, 0, STATE_COPY_SOURCE, depth);
                            }
                            DepthSyncKind::BufferThrough => {
                                let twin_raw = twin.as_raw() as usize;
                                transition_plane_subs(list, res, true, per_plane, 0, depth, Some(STATE_COPY_SOURCE));
                                list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
                                let _ = copy_plane_through_buffer(list, &dev, &twin, 0, depth_res, 0, per_plane, depth);
                                list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                                transition_plane_from(list, res, true, per_plane, 0, STATE_COPY_SOURCE, depth);
                            }
                            DepthSyncKind::FullscreenPass { twin_srv_fmt, twin_uav_fmt } => {
                                let heaps = {
                                    let mut g = ENTRIES.lock().unwrap();
                                    g.as_mut().and_then(|m| m.get_mut(&res)).and_then(|e| ensure_pass_heaps(&dev, depth_res, &twin, e, twin_srv_fmt, twin_uav_fmt))
                                };
                                if let (Some((_, _, cs_rs, cs_pso)), Some((_, srv_heap))) = (get_pipelines(&dev), heaps) {
                                    transition_plane_subs(list, res, true, per_plane, 0, depth, Some(STATE_SRV_ALL));
                                    let srv_raw = srv_heap.as_raw() as usize;
                                    let set_heaps: unsafe extern "system" fn(*mut c_void, u32, *const usize) = std::mem::transmute(real::orig_fn(list, 28));
                                    let set_cs_rs: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list, 29));
                                    let set_pso: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list, 25));
                                    let set_cs_table: unsafe extern "system" fn(*mut c_void, u32, u64) = std::mem::transmute(real::orig_fn(list, 31));
                                    let set_cs_consts: unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32) = std::mem::transmute(real::orig_fn(list, 35));
                                    let dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32) = std::mem::transmute(real::orig_fn(list, 14));

                                    set_heaps(list, 1, &srv_raw);
                                    set_cs_rs(list, cs_rs.as_raw());
                                    set_pso(list, cs_pso.as_raw());
                                    let gpu_base = srv_heap.GetGPUDescriptorHandleForHeapStart().ptr;
                                    let inc = inc_size(dev_raw, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as u64;
                                    set_cs_table(list, 1, gpu_base + inc);
                                    let mips_u32 = mip_levels as u32;
                                    let mut dispatched_mips = vec![false; mips_u32 as usize];
                                    for &(local_sub, _, _) in depth {
                                        let m = local_sub % mips_u32;
                                        if dispatched_mips[m as usize] {
                                            continue;
                                        }
                                        dispatched_mips[m as usize] = true;
                                        let mw = ((width as u32) >> m).max(1);
                                        let mh = (height >> m).max(1);
                                        let c = [mw, mh, array_size as u32, m];
                                        set_cs_consts(list, 0, 4, c.as_ptr(), 0);
                                        set_cs_table(list, 2, gpu_base + (2 + m as u64) * inc);
                                        dispatch(list, mw.div_ceil(8), mh.div_ceil(8), array_size as u32);
                                    }
                                    // Order internal compute UAV writes to `twin` before subsequent title dispatches.
                                    let twin_raw = twin.as_raw() as usize;
                                    let uav_barrier = D3D12_RESOURCE_BARRIER {
                                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                                pResource: std::mem::transmute_copy(&twin_raw),
                                            }),
                                        },
                                    };
                                    list_x::pc_resource_barriers(list, &[uav_barrier]);
                                    transition_plane_from(list, res, true, per_plane, 0, STATE_SRV_ALL, depth);
                                    clobbered_cs = true;
                                } else {
                                    transition_plane_subs(list, res, true, per_plane, 0, depth, None);
                                }
                            }
                        }
                    } else {
                        transition_plane_subs(list, res, plan.two_plane, per_plane, 0, depth, None);
                    }
                }

                if plan.two_plane && !stencil.is_empty() {
                    if let Some(twin) = stencil_twin {
                        let twin_raw = twin.as_raw() as usize;
                        transition_plane_subs(list, res, true, per_plane, 1, stencil, Some(STATE_COPY_SOURCE));
                        list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
                        let _ = copy_plane_through_buffer(list, &dev, &twin, 0, depth_res, 1, per_plane, stencil);
                        list_x::transition(list, twin_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                        transition_plane_from(list, res, true, per_plane, 1, STATE_COPY_SOURCE, stencil);
                    } else {
                        transition_plane_subs(list, res, true, per_plane, 1, stencil, None);
                    }
                }
            }
        }
    }

    if clobbered_gfx || clobbered_cs {
        list_x::restore_after_internal_pass(list, clobbered_gfx, clobbered_cs);
    }
}

// ---------------------------------------------------------------------------------------
// LOCAL agent I: copy at use, not at barrier (`XCL_LOCAL_TWIN_AT_USE=1`).
// The title orders its compute writes with enhanced barriers, which reach XCL as one global UAV
// barrier naming no resource: the dirty marks of #1989 (resource UAV barriers, legacy
// transitions) never fire. Here a twin is taken as newer than its depth-stencil from every
// dispatch (direct or indirect) issued while the twin exists (it exists only once a UAV view of
// it was made), and is copied into the depth-stencil before each DSV bind of it, before copies
// touching it and at the close of each DIRECT list. A DSV bind marks the depth-stencil as newer
// (rasterisation may write it); the next dispatch on a DIRECT list first copies it into the twin,
// unless the twin is newer itself (then the twin wins, counted as a conflict). Entries narrower
// than `XCL_LOCAL_TWIN_MINW` (default 1920) are left to the rules of #1989.
// ---------------------------------------------------------------------------------------

pub(super) fn local_at_use() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_LOCAL_TWIN_AT_USE").is_some())
}

/// LOCAL agent I: any of this agent's switches
fn local_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| local_at_use() || std::env::var_os("XCL_LOCAL_TWIN_SNAP_CS").is_some())
}

fn local_minw() -> u64 {
    static W: OnceLock<u64> = OnceLock::new();
    *W.get_or_init(|| std::env::var("XCL_LOCAL_TWIN_MINW").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1920))
}

fn local_marked(e: &Entry) -> bool {
    e.width >= local_minw() && (e.depth_twin.is_some() || e.stencil_twin.is_some())
}

// counters: 0 dispatches seen, 1 depth->twin ops, 2 twin->depth at DSV bind, 3 at list close,
// 4 at copy, 5 conflicts (both newer), 6 depth newer on a non-DIRECT list (twin taken as newer),
// 7 dispatches on non-DIRECT lists
static LSTAT: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn local_stat(k: usize) {
    let n = LSTAT[k].fetch_add(1, Ordering::Relaxed) + 1;
    if (k != 0 && n >= 64 && n.is_power_of_two()) || (k == 0 && n % 20000 == 0) {
        let v: Vec<u64> = LSTAT.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        crate::xlog_info!(
            "LOCAL twin-at-use: dispatches {}, depth->twin {}, twin->depth at DSV bind {} / list close {} / copy {}, conflicts {}, depth newer on non-direct list {}, non-direct dispatches {}",
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]
        );
    }
}

fn all_subs(states: &[u32]) -> Vec<(u32, u32, u32)> {
    states.iter().enumerate().map(|(s, &st)| (s as u32, st, st)).collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LocalUse {
    DsvBind,
    RasterWrite,
    Close,
    CopySrc,
    CopyDst,
}

/// LOCAL agent I: before a dispatch (Dispatch or ExecuteIndirect with a dispatch signature).
pub(super) unsafe fn local_before_dispatch(list: *mut c_void) {
    if !local_at_use() || !any() {
        return;
    }
    let is_direct = list_x::tracked_list_type(list).is_none_or(|t| t == D3D12_COMMAND_LIST_TYPE_DIRECT);
    local_stat(0);
    if !is_direct {
        local_stat(7);
    }
    let mut ops = Vec::new();
    {
        let mut g = ENTRIES.lock().unwrap();
        let Some(m) = g.as_mut() else { return };
        for (&res, e) in m.iter_mut() {
            if !local_marked(e) {
                continue;
            }
            let has_d = e.depth_twin.is_some();
            let has_s = e.plan.two_plane && e.stencil_twin.is_some();
            let need_d = has_d && e.depth_plane_dirty && !e.depth_twin_dirty;
            let need_s = has_s && e.stencil_plane_dirty && !e.stencil_twin_dirty;
            if (has_d && e.depth_plane_dirty && e.depth_twin_dirty) || (has_s && e.stencil_plane_dirty && e.stencil_twin_dirty) {
                local_stat(5);
            }
            if !is_direct && ((has_d && e.depth_plane_dirty) || (has_s && e.stencil_plane_dirty)) {
                // run i3: taking the twin as newer here copied a stale (zero) twin over the
                // rasterised main depth at the next list close: the depth-stencil is kept instead
                local_stat(6);
                continue;
            }
            if need_d || need_s {
                if is_direct {
                    ops.push(SyncOp::DepthToTwin {
                        res,
                        depth: if need_d { all_subs(&e.pc_states_depth) } else { Vec::new() },
                        stencil: if need_s { all_subs(&e.pc_states_stencil) } else { Vec::new() },
                    });
                    local_stat(1);
                } else {
                    local_stat(6);
                }
            }
            if has_d {
                e.depth_twin_dirty = true;
                e.depth_plane_dirty = false;
            }
            if has_s {
                e.stencil_twin_dirty = true;
                e.stencil_plane_dirty = false;
            }
        }
    }
    if !ops.is_empty() {
        execute_syncs(list, &ops);
    }
}

/// LOCAL agent I: twin -> depth-stencil of `res` (both planes, where newer) before a use of it on
/// a DIRECT list; after a DSV bind or as a copy destination the depth-stencil is taken as newer.
pub(super) unsafe fn local_sync_at_use(list: *mut c_void, res: usize, why: LocalUse) {
    if !local_at_use() || !any() {
        return;
    }
    let is_direct = list_x::tracked_list_type(list).is_none_or(|t| t == D3D12_COMMAND_LIST_TYPE_DIRECT);
    let op = {
        let mut g = ENTRIES.lock().unwrap();
        let Some(e) = g.as_mut().and_then(|m| m.get_mut(&res)) else {
            return;
        };
        if !local_marked(e) || !is_direct {
            return;
        }
        let d = e.depth_twin_dirty && e.depth_twin.is_some();
        let s = e.plan.two_plane && e.stencil_twin_dirty && e.stencil_twin.is_some();
        let op = (d || s).then(|| SyncOp::TwinToDepth {
            res,
            depth: if d { all_subs(&e.pc_states_depth) } else { Vec::new() },
            stencil: if s { all_subs(&e.pc_states_stencil) } else { Vec::new() },
        });
        if d {
            e.depth_twin_dirty = false;
        }
        if s {
            e.stencil_twin_dirty = false;
        }
        if matches!(why, LocalUse::RasterWrite | LocalUse::CopyDst) {
            e.depth_plane_dirty = e.depth_twin.is_some();
            e.stencil_plane_dirty = e.plan.two_plane && e.stencil_twin.is_some();
        }
        if op.is_some() {
            static L: AtomicU64 = AtomicU64::new(0);
            let n = L.fetch_add(1, Ordering::Relaxed);
            if n < 8 || n.is_power_of_two() {
                let w = match why {
                    LocalUse::DsvBind => "DSV bind",
                    LocalUse::RasterWrite => "depth/stencil-writing pipeline with the DSV bound",
                    LocalUse::Close => "list close",
                    LocalUse::CopySrc => "copy (source)",
                    LocalUse::CopyDst => "copy (destination)",
                };
                crate::xlog_info!("LOCAL twin-at-use: twin -> depth-stencil of {res:#x} ({}x{}) planes depth {d} stencil {s} at {w} (#{})", e.width, e.height, n + 1);
            }
        }
        op
    };
    if let Some(op) = op {
        local_stat(match why {
            LocalUse::DsvBind | LocalUse::RasterWrite => 2,
            LocalUse::Close => 3,
            _ => 4,
        });
        execute_syncs(list, &[op]);
    }
}

/// LOCAL agent I: pipelines known NOT to write depth or stencil (translated graphics pipelines
/// with depth write mask 0 and stencil writes off); any other pipeline is taken as writing.
static LOCAL_NO_DS_WRITE: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

pub(super) fn local_note_pso_ds_write(pso: usize, writes: bool) {
    if !local_at_use() || pso == 0 {
        return;
    }
    let mut g = LOCAL_NO_DS_WRITE.lock().unwrap();
    let s = g.get_or_insert_with(Default::default);
    if writes {
        s.remove(&pso);
    } else {
        s.insert(pso);
    }
}

fn local_pso_writes(pso: usize) -> bool {
    if pso == 0 {
        return false;
    }
    let real = super::pso_async::resolve(pso);
    let g = LOCAL_NO_DS_WRITE.lock().unwrap();
    !g.as_ref().is_some_and(|s| s.contains(&pso) || s.contains(&real))
}

/// LOCAL agent I: after SetPipelineState / OMSetRenderTargets: a pipeline that may write depth or
/// stencil with a tracked DSV bound makes the depth-stencil newer than its twin (twin first copied
/// in if it was newer).
pub(super) unsafe fn local_raster_check(list: *mut c_void, pso: usize) {
    if !local_at_use() || !any() {
        return;
    }
    let Some(dsv) = list_x::bound_dsv(list) else { return };
    let Some(res) = DSV_TO_RES.lock().unwrap().as_ref().and_then(|m| m.get(&dsv).copied()) else {
        return;
    };
    if local_pso_writes(pso) {
        local_sync_at_use(list, res, LocalUse::RasterWrite);
    }
}

/// LOCAL agent I: before Close of a DIRECT list: every newer twin into its depth-stencil.
pub(super) unsafe fn local_before_close(list: *mut c_void) {
    if !local_at_use() || !any() {
        return;
    }
    let dirty: Vec<usize> = {
        let g = ENTRIES.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        m.iter().filter(|(_, e)| local_marked(e) && (e.depth_twin_dirty || e.stencil_twin_dirty)).map(|(&r, _)| r).collect()
    };
    for res in dirty {
        local_sync_at_use(list, res, LocalUse::Close);
    }
}

/// LOCAL agent I: `res` is the source (or destination) of a copy about to be recorded.
pub(super) unsafe fn local_before_copy(list: *mut c_void, res: usize, dst: bool) {
    if local_at_use() && res != 0 && any() {
        local_sync_at_use(list, res, if dst { LocalUse::CopyDst } else { LocalUse::CopySrc });
    }
}

/// LOCAL agent I (`XCL_LOCAL_TWIN_SNAP_CS=<cs hash prefixes>`): after a dispatch of one of those
/// programs, the twins of the screen-sized entries are copied into snapshot textures (dumped with
/// the targets, `local_dump_extra`): what the compute program left in them.
pub(super) unsafe fn local_after_dispatch(list: *mut c_void, pso: usize) {
    static PRE: OnceLock<Vec<String>> = OnceLock::new();
    let pre = PRE.get_or_init(|| std::env::var("XCL_LOCAL_TWIN_SNAP_CS").map(|v| v.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect()).unwrap_or_default());
    if pre.is_empty() || !any() || !super::rtsum::on() {
        return;
    }
    let sh = super::rtsum::shaders(pso);
    let Some(h) = sh.strip_prefix("cs ") else { return };
    if !pre.iter().any(|p| h.starts_with(p.as_str())) {
        return;
    }
    let Some(dev) = list_x::device_of_list(list) else { return };
    let mut jobs: Vec<(usize, ID3D12Resource, usize)> = Vec::new();
    {
        let mut g = ENTRIES.lock().unwrap();
        let Some(m) = g.as_mut() else { return };
        for (&res, e) in m.iter_mut() {
            if e.width < 1920 {
                continue;
            }
            for k in 0..2 {
                let Some(twin) = (if k == 0 { e.depth_twin.clone() } else { e.stencil_twin.clone() }) else { continue };
                if e.local_snap[k].is_none() {
                    let mut d = twin.GetDesc();
                    d.Flags = D3D12_RESOURCE_FLAG_NONE;
                    let mut out: Option<ID3D12Resource> = None;
                    if dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut out).is_err() {
                        continue;
                    }
                    e.local_snap[k] = out;
                }
                if let Some(s) = e.local_snap[k].as_ref() {
                    jobs.push((twin.as_raw() as usize, s.clone(), res));
                }
            }
        }
    }
    static N: AtomicU64 = AtomicU64::new(0);
    for (twin, snap, res) in jobs {
        list_x::transition(list, twin, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list_x::pc_copy_resource(list, snap.as_raw() as usize, twin);
        list_x::transition(list, twin, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 8 || n.is_power_of_two() {
            crate::xlog_info!(
                "LOCAL twin-snap: after cs {h} on list {list:p} (type {:?}): twin {twin:#x} of {res:#x} -> snapshot {:p} (#{})",
                list_x::tracked_list_type(list).map(|t| t.0),
                snap.as_raw(),
                n + 1
            );
        }
    }
}

/// LOCAL agent I: resources to dump with the targets (twins and snapshots of screen-sized entries),
/// each logged with its role.
pub(super) fn local_dump_extra(frame: u64) -> Vec<usize> {
    if !local_on() || !any() {
        return Vec::new();
    }
    let g = ENTRIES.lock().unwrap();
    let Some(m) = g.as_ref() else { return Vec::new() };
    let mut v = Vec::new();
    for (&res, e) in m.iter() {
        if e.width < 1920 {
            continue;
        }
        let r = |o: &Option<ID3D12Resource>| o.as_ref().map_or(0, |x| x.as_raw() as usize);
        let roles = [(r(&e.depth_twin), "depth twin"), (r(&e.stencil_twin), "stencil twin"), (r(&e.local_snap[0]), "depth snapshot"), (r(&e.local_snap[1]), "stencil snapshot")];
        for (p, role) in roles {
            if p != 0 {
                crate::xlog_info!(
                    "LOCAL twin-dump: frame {frame}: {p:#x} = {role} of {res:#x} ({}x{}, plan {:?}; twin newer {} / {}, depth newer {} / {})",
                    e.width,
                    e.height,
                    e.plan.depth_sync,
                    e.depth_twin_dirty,
                    e.stencil_twin_dirty,
                    e.depth_plane_dirty,
                    e.stencil_plane_dirty
                );
                v.push(p);
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twin_plans_cover_console_depth_formats() {
        let d32 = plan_for_console_format(40).expect("D32_FLOAT");
        assert_eq!((d32.depth_twin_fmt, d32.stencil_twin_fmt, d32.depth_sync, d32.two_plane), (39, None, DepthSyncKind::DirectCopy, false));

        let d16 = plan_for_console_format(55).expect("D16_UNORM");
        assert_eq!((d16.depth_twin_fmt, d16.stencil_twin_fmt, d16.depth_sync, d16.two_plane), (53, None, DepthSyncKind::DirectCopy, false));

        let d32s8 = plan_for_console_format(20).expect("D32_FLOAT_S8X24_UINT");
        assert_eq!((d32s8.depth_twin_fmt, d32s8.stencil_twin_fmt, d32s8.depth_sync, d32s8.two_plane), (39, Some(60), DepthSyncKind::BufferThrough, true));

        let d16s8 = plan_for_console_format(118).expect("D16_UNORM_S8_UINT");
        assert_eq!(d16s8.depth_twin_fmt, 53);
        assert_eq!(d16s8.stencil_twin_fmt, Some(60));
        assert!(d16s8.two_plane);
        assert_eq!(d16s8.depth_sync, DepthSyncKind::FullscreenPass { twin_srv_fmt: 56, twin_uav_fmt: 56 });

        let d24s8 = plan_for_console_format(45).expect("D24_UNORM_S8_UINT");
        assert_eq!(d24s8.depth_twin_fmt, 39);
        assert_eq!(d24s8.stencil_twin_fmt, Some(60));
        assert_eq!(d24s8.depth_sync, DepthSyncKind::FullscreenPass { twin_srv_fmt: 41, twin_uav_fmt: 41 });

        assert!(plan_for_console_format(28).is_none(), "color format has no depth twin plan");
    }

    #[test]
    fn uav_format_and_plane_classification() {
        // Stencil plane detection via format or explicit PlaneSlice == 1
        assert!(is_stencil_uav_format(62, 0)); // R8_UINT
        assert!(is_stencil_uav_format(120, 0)); // console X16_TYPELESS_G8_UINT
        assert!(is_stencil_uav_format(47, 0)); // X24_TYPELESS_G8_UINT
        assert!(is_stencil_uav_format(22, 0)); // X32_TYPELESS_G8X24_UINT
        assert!(is_stencil_uav_format(0, 1)); // explicit PlaneSlice 1
        assert!(!is_stencil_uav_format(56, 0)); // R16_UNORM depth
        assert!(!is_stencil_uav_format(41, 0)); // R32_FLOAT depth

        // Legal typed UAV formats on each typeless twin
        assert_eq!(twin_uav_format(60, 120), 62);
        assert_eq!(twin_uav_format(60, 61), 61);
        assert_eq!(twin_uav_format(53, 118), 56);
        assert_eq!(twin_uav_format(53, 57), 57);
        assert_eq!(twin_uav_format(39, 40), 41);
        assert_eq!(twin_uav_format(39, 42), 42);

        // Subresource plane split (2 mips * 3 slices = 6 subresources per plane)
        assert_eq!(plane_for_subresource(D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, 2, 3, true), PlaneMask::Both);
        assert_eq!(plane_for_subresource(0, 2, 3, true), PlaneMask::Depth);
        assert_eq!(plane_for_subresource(5, 2, 3, true), PlaneMask::Depth);
        assert_eq!(plane_for_subresource(6, 2, 3, true), PlaneMask::Stencil);
        assert_eq!(plane_for_subresource(11, 2, 3, true), PlaneMask::Stencil);
        assert_eq!(plane_for_subresource(D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, 2, 3, false), PlaneMask::Depth);
    }

    #[test]
    fn state_machine_handles_split_and_async_compute_transitions() {
        let plan = plan_for_console_format(118).unwrap();
        let mut e = Entry::new(0x1, plan, 1920, 1080, 1, 1, STATE_DEPTH_WRITE, true);

        // Without twins allocated yet, leaving UAV becomes a plain TransitionOnly (DEPTH_WRITE -> DEPTH_READ)
        let (pass, uav, op) = decide_transition(&mut e, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, STATE_UAV, 0x20, 0, true);
        assert!(!pass && uav);
        assert_eq!(
            op,
            Some(SyncOp::TransitionOnly {
                res: 0,
                depth: vec![(0, STATE_DEPTH_WRITE, 0x20)],
                stencil: vec![(0, STATE_DEPTH_WRITE, 0x20)],
            })
        );
        assert_eq!((e.pc_states_depth[0], e.pc_states_stencil[0]), (0x20, 0x20));

        // Split barrier BEGIN_ONLY touching UAV is suppressed, and END_ONLY performs the atomic transition
        let (pass, uav, op) = decide_transition(&mut e, 1, 0x20, STATE_UAV, 1, true);
        assert!(!pass && !uav && op.is_none());
        assert!(e.split_uav_stencil[0] && !e.split_uav_depth[0]);

        let (pass, uav, op) = decide_transition(&mut e, 1, 0x20, STATE_UAV, 2, true);
        assert!(!pass && uav);
        assert!(!e.split_uav_stencil[0]);
        assert_eq!(
            op,
            Some(SyncOp::TransitionOnly {
                res: 0,
                depth: vec![],
                stencil: vec![(0, 0x20, STATE_DEPTH_WRITE)],
            })
        );
        // Plane 0 stayed in DEPTH_READ while plane 1 moved to DEPTH_WRITE
        assert_eq!((e.pc_states_depth[0], e.pc_states_stencil[0]), (0x20, STATE_DEPTH_WRITE));

        // Multi-mip per-subresource tracking: transitioning only mip 1 must leave mip 0 untouched
        // and entering UAV must use each subresource's tracked PC state as StateBefore (not title `before`).
        let mut multi = Entry::new(0x1, plan, 512, 512, 1, 2, STATE_DEPTH_WRITE, false);
        let (pass, _, op) = decide_transition(&mut multi, 1, STATE_DEPTH_WRITE, 0x20, 0, true);
        assert!(pass && op.is_none());
        assert_eq!(multi.pc_states_depth, vec![STATE_DEPTH_WRITE, 0x20]);
        let (pass, uav, op) = decide_transition(&mut multi, 1, STATE_COMMON, STATE_UAV, 0, true);
        assert!(!pass && uav);
        assert_eq!(
            op,
            Some(SyncOp::TransitionOnly {
                res: 0,
                depth: vec![(1, 0x20, STATE_DEPTH_WRITE)],
                stencil: vec![],
            }),
            "single-subresource UAV entry transitions only mip 1 from its tracked PC state 0x20"
        );
        assert_eq!(multi.pc_states_depth, vec![STATE_DEPTH_WRITE, STATE_DEPTH_WRITE]);

        // On a COMPUTE list, UAV transitions are suppressed with a global UAV barrier (no invalid DEPTH state)
        let (pass, uav, op) = decide_transition(&mut e, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, STATE_UAV, STATE_COMMON, 0, false);
        assert!(!pass && uav && op.is_none());
    }

    #[test]
    fn uav_barrier_returns_zero_for_untracked_resources() {
        assert_eq!(on_uav_barrier(0), [0, 0]);
        assert_eq!(on_uav_barrier(0xDEAD_BEE0), [0, 0]);
    }
}
