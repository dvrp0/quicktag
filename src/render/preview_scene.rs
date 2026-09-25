#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraParameters {
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
    pub scale: f32,
    pub pan: [f32; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreviewScene {
    pub camera: CameraParameters,
    pub output_size: [u32; 2],
    pub exposure: f32,
    pub environment_rotation_degrees: f32,
    pub background: [f32; 4],
}

impl PreviewScene {
    pub fn revamp_br33_baseline() -> Self {
        Self {
            camera: CameraParameters {
                yaw_degrees: -24.0,
                pitch_degrees: 14.0,
                scale: 3.1,
                pan: [0.0, 0.0],
            },
            output_size: [1024, 640],
            exposure: 1.0,
            environment_rotation_degrees: 0.0,
            background: [0.005, 0.009, 0.018, 1.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_scene_is_stable() {
        let scene = PreviewScene::revamp_br33_baseline();
        assert_eq!(scene.output_size, [1024, 640]);
        assert_eq!(scene.camera.yaw_degrees, -24.0);
        assert_eq!(scene.camera.pitch_degrees, 14.0);
    }
}
