#![cfg(target_os = "windows")]
use std::mem::MaybeUninit;

use crate::structs::{XgResourceLayout, XgTexture2DDesc};

pub mod structs;

#[link(name = "xg", kind = "raw-dylib")]
unsafe extern "C" {
    fn XGCreateTexture2DComputer(
        desc: *const XgTexture2DDesc,
        out: *mut *mut std::ffi::c_void,
    ) -> i32;
}

pub struct XgTexture2DComputer {
    ptr: *mut std::ffi::c_void,
}

impl XgTexture2DComputer {
    pub fn new(desc: &XgTexture2DDesc) -> Result<Self, i32> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let hr = unsafe { XGCreateTexture2DComputer(desc, &mut ptr) };
        if hr < 0 {
            return Err(hr);
        }
        Ok(Self { ptr })
    }

    pub fn get_resource_layout(&self) -> Result<XgResourceLayout, i32> {
        let mut layout = MaybeUninit::<XgResourceLayout>::uninit();
        let hr = (self.vt().get_resource_layout)(self.ptr, layout.as_mut_ptr());
        if hr < 0 {
            return Err(hr);
        }
        Ok(unsafe { layout.assume_init() })
    }

    pub fn get_texel_element_offset_bytes(
        &self,
        subresource: u32,
        level: u32,
        x: u64,
        y: u32,
        array_index: u32,
        element: u32,
    ) -> i64 {
        (self.vt().get_texel_element_offset_bytes)(
            self.ptr,
            subresource,
            level,
            x,
            y,
            array_index,
            element,
        )
    }

    fn vt(&self) -> &XgTextureComputerVtable {
        unsafe { &*(self.ptr as *const *const XgTextureComputerVtable).read() }
    }
}

// https://github.com/Gravemind2401/Reclaimer/blob/master/Reclaimer.Blam/Utilities/XG.cs#L284
#[repr(C)]
struct XgTextureComputerVtable {
    add_ref: extern "C" fn(*mut std::ffi::c_void) -> u32,
    release: extern "C" fn(*mut std::ffi::c_void) -> u32,
    get_resource_layout: extern "C" fn(*mut std::ffi::c_void, *mut XgResourceLayout) -> i32,
    get_resource_size_bytes: extern "C" fn(*mut std::ffi::c_void) -> u64,
    get_resource_base_alignment_bytes: extern "C" fn(*mut std::ffi::c_void) -> u64,
    get_mip_level_offset_bytes: extern "C" fn(*mut std::ffi::c_void, u32, u32) -> u64,
    get_texel_element_offset_bytes:
        extern "C" fn(*mut std::ffi::c_void, u32, u32, u64, u32, u32, u32) -> i64,
    get_texel_coordinate: extern "C" fn(
        *mut std::ffi::c_void,
        u64,
        *mut u32,
        *mut u32,
        *mut u64,
        *mut u32,
        *mut u32,
        *mut u32,
    ) -> i32,
}
