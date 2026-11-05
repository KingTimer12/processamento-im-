use std::f32::consts::{PI, TAU};

use glam::{Mat3, Mat4, Vec3};

pub(crate) const MESH_LAYOUT: [i32; 2] = [3, 3];

#[derive(Clone, Default)]
pub struct MeshBuilder {
    data: Vec<f32>,
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vertex_count(&self) -> usize {
        self.data.len() / 6
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    fn push(&mut self, p: Vec3, n: Vec3) {
        self.data.extend_from_slice(&[p.x, p.y, p.z, n.x, n.y, n.z]);
    }

    pub fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3) -> &mut Self {
        let n = (b - a).cross(c - a).normalize_or_zero();
        if n != Vec3::ZERO {
            self.push(a, n);
            self.push(b, n);
            self.push(c, n);
        }
        self
    }

    pub fn triangle_smooth(&mut self, v: [(Vec3, Vec3); 3]) -> &mut Self {
        for (p, n) in v {
            self.push(p, n);
        }
        self
    }

    pub fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> &mut Self {
        self.triangle(a, b, c).triangle(a, c, d)
    }

    pub fn polygon(&mut self, points: &[Vec3]) -> &mut Self {
        for i in 1..points.len().saturating_sub(1) {
            self.triangle(points[0], points[i], points[i + 1]);
        }
        self
    }

    pub fn flatten(&mut self) -> &mut Self {
        for tri in self.data.chunks_exact_mut(18) {
            let p = |k: usize| Vec3::new(tri[k * 6], tri[k * 6 + 1], tri[k * 6 + 2]);
            let n = (p(1) - p(0)).cross(p(2) - p(0)).normalize_or_zero();
            if n == Vec3::ZERO {
                continue;
            }
            for k in 0..3 {
                tri[k * 6 + 3..k * 6 + 6].copy_from_slice(&[n.x, n.y, n.z]);
            }
        }
        self
    }

    pub fn transform(&mut self, m: Mat4) -> &mut Self {
        let normal_m = Mat3::from_mat4(m.inverse().transpose());
        for v in self.data.chunks_exact_mut(6) {
            let p = m.transform_point3(Vec3::new(v[0], v[1], v[2]));
            let n = (normal_m * Vec3::new(v[3], v[4], v[5])).normalize_or_zero();
            v.copy_from_slice(&[p.x, p.y, p.z, n.x, n.y, n.z]);
        }
        self
    }

    pub fn append(&mut self, other: &MeshBuilder) -> &mut Self {
        self.data.extend_from_slice(&other.data);
        self
    }

    fn solid_triangle(&mut self, a: Vec3, mut b: Vec3, mut c: Vec3) {
        let n = (b - a).cross(c - a);
        if n.dot(a + b + c) < 0.0 {
            std::mem::swap(&mut b, &mut c);
        }
        self.triangle(a, b, c);
    }

    fn solid_quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        self.solid_triangle(a, b, c);
        self.solid_triangle(a, c, d);
    }

    pub fn plane(width: f32, depth: f32) -> Self {
        let (w, d) = (width / 2.0, depth / 2.0);
        let mut m = Self::new();
        m.quad(
            Vec3::new(-w, 0.0, -d),
            Vec3::new(-w, 0.0, d),
            Vec3::new(w, 0.0, d),
            Vec3::new(w, 0.0, -d),
        );
        m
    }

    pub fn cube(size: f32) -> Self {
        let h = size / 2.0;
        let p = |x: f32, y: f32, z: f32| Vec3::new(x * h, y * h, z * h);
        let mut m = Self::new();
        m.solid_quad(p(-1., -1., -1.), p(1., -1., -1.), p(1., 1., -1.), p(-1., 1., -1.));
        m.solid_quad(p(-1., -1., 1.), p(1., -1., 1.), p(1., 1., 1.), p(-1., 1., 1.));
        m.solid_quad(p(-1., -1., -1.), p(-1., 1., -1.), p(-1., 1., 1.), p(-1., -1., 1.));
        m.solid_quad(p(1., -1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(1., -1., 1.));
        m.solid_quad(p(-1., -1., -1.), p(1., -1., -1.), p(1., -1., 1.), p(-1., -1., 1.));
        m.solid_quad(p(-1., 1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(-1., 1., 1.));
        m
    }

    pub fn regular_polygon(sides: u32, radius: f32) -> Self {
        let points: Vec<Vec3> = (0..sides.max(3))
            .map(|i| {
                let a = TAU * i as f32 / sides.max(3) as f32;
                Vec3::new(radius * a.cos(), radius * a.sin(), 0.0)
            })
            .collect();
        let mut m = Self::new();
        m.polygon(&points);
        m
    }

    pub fn prism(sides: u32, radius: f32, height: f32) -> Self {
        let sides = sides.max(3);
        let (top, bottom) = (ring(sides, radius, height / 2.0), ring(sides, radius, -height / 2.0));
        let (ct, cb) = (Vec3::Y * height / 2.0, Vec3::NEG_Y * height / 2.0);
        let mut m = Self::new();
        for i in 0..sides as usize {
            let j = (i + 1) % sides as usize;
            m.solid_quad(bottom[i], bottom[j], top[j], top[i]);
            m.solid_triangle(ct, top[i], top[j]);
            m.solid_triangle(cb, bottom[i], bottom[j]);
        }
        m
    }

    pub fn pyramid(sides: u32, radius: f32, height: f32) -> Self {
        let sides = sides.max(3);
        let base = ring(sides, radius, -height / 2.0);
        let (apex, cb) = (Vec3::Y * height / 2.0, Vec3::NEG_Y * height / 2.0);
        let mut m = Self::new();
        for i in 0..sides as usize {
            let j = (i + 1) % sides as usize;
            m.solid_triangle(base[i], base[j], apex);
            m.solid_triangle(cb, base[i], base[j]);
        }
        m
    }

    pub fn cylinder(radius: f32, height: f32, segments: u32) -> Self {
        let segments = segments.max(3);
        let (top, bottom) = (ring(segments, radius, height / 2.0), ring(segments, radius, -height / 2.0));
        let (ct, cb) = (Vec3::Y * height / 2.0, Vec3::NEG_Y * height / 2.0);
        let mut m = Self::new();
        for i in 0..segments as usize {
            let j = (i + 1) % segments as usize;
            let (ni, nj) = (radial(i, segments), radial(j, segments));
            m.triangle_smooth([(bottom[i], ni), (bottom[j], nj), (top[j], nj)]);
            m.triangle_smooth([(bottom[i], ni), (top[j], nj), (top[i], ni)]);
            m.solid_triangle(ct, top[i], top[j]);
            m.solid_triangle(cb, bottom[i], bottom[j]);
        }
        m
    }

    pub fn cone(radius: f32, height: f32, segments: u32) -> Self {
        let segments = segments.max(3);
        let base = ring(segments, radius, -height / 2.0);
        let (apex, cb) = (Vec3::Y * height / 2.0, Vec3::NEG_Y * height / 2.0);
        let slope = |a: f32| Vec3::new(height * a.cos(), radius, height * a.sin()).normalize();
        let mut m = Self::new();
        for i in 0..segments as usize {
            let j = (i + 1) % segments as usize;
            let (a0, a1) = (TAU * i as f32 / segments as f32, TAU * (i + 1) as f32 / segments as f32);
            m.triangle_smooth([(base[i], slope(a0)), (base[j], slope(a1)), (apex, slope((a0 + a1) / 2.0))]);
            m.solid_triangle(cb, base[i], base[j]);
        }
        m
    }

    pub fn sphere(radius: f32, segments: u32) -> Self {
        let segments = segments.max(3);
        let rings = (segments / 2).max(2);
        let point = |i: u32, j: u32| {
            let (theta, phi) = (PI * i as f32 / rings as f32, TAU * j as f32 / segments as f32);
            let n = Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
            (n * radius, n)
        };
        let mut m = Self::new();
        for i in 0..rings {
            for j in 0..segments {
                let (p00, p01) = (point(i, j), point(i, j + 1));
                let (p10, p11) = (point(i + 1, j), point(i + 1, j + 1));
                if i != rings - 1 {
                    m.triangle_smooth([p00, p10, p11]);
                }
                if i != 0 {
                    m.triangle_smooth([p00, p11, p01]);
                }
            }
        }
        m
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Cube { size: f32 },
    Plane { width: f32, depth: f32 },
    RegularPolygon { sides: u32, radius: f32 },
    Prism { sides: u32, radius: f32, height: f32 },
    Pyramid { sides: u32, radius: f32, height: f32 },
    Cylinder { radius: f32, height: f32, segments: u32 },
    Cone { radius: f32, height: f32, segments: u32 },
    Sphere { radius: f32, segments: u32 },
}

impl Shape {
    pub fn build(&self) -> MeshBuilder {
        match *self {
            Shape::Cube { size } => MeshBuilder::cube(size),
            Shape::Plane { width, depth } => MeshBuilder::plane(width, depth),
            Shape::RegularPolygon { sides, radius } => MeshBuilder::regular_polygon(sides, radius),
            Shape::Prism { sides, radius, height } => MeshBuilder::prism(sides, radius, height),
            Shape::Pyramid { sides, radius, height } => MeshBuilder::pyramid(sides, radius, height),
            Shape::Cylinder { radius, height, segments } => MeshBuilder::cylinder(radius, height, segments),
            Shape::Cone { radius, height, segments } => MeshBuilder::cone(radius, height, segments),
            Shape::Sphere { radius, segments } => MeshBuilder::sphere(radius, segments),
        }
    }

    pub fn segments(&self) -> Option<u32> {
        match *self {
            Shape::RegularPolygon { sides, .. }
            | Shape::Prism { sides, .. }
            | Shape::Pyramid { sides, .. } => Some(sides),
            Shape::Cylinder { segments, .. }
            | Shape::Cone { segments, .. }
            | Shape::Sphere { segments, .. } => Some(segments),
            Shape::Cube { .. } | Shape::Plane { .. } => None,
        }
    }

    pub fn with_segments(mut self, n: u32) -> Self {
        match &mut self {
            Shape::RegularPolygon { sides, .. }
            | Shape::Prism { sides, .. }
            | Shape::Pyramid { sides, .. } => *sides = n,
            Shape::Cylinder { segments, .. }
            | Shape::Cone { segments, .. }
            | Shape::Sphere { segments, .. } => *segments = n,
            Shape::Cube { .. } | Shape::Plane { .. } => {}
        }
        self
    }
}

fn ring(sides: u32, radius: f32, y: f32) -> Vec<Vec3> {
    (0..sides)
        .map(|i| {
            let a = TAU * i as f32 / sides as f32;
            Vec3::new(radius * a.cos(), y, radius * a.sin())
        })
        .collect()
}

fn radial(i: usize, sides: u32) -> Vec3 {
    let a = TAU * i as f32 / sides as f32;
    Vec3::new(a.cos(), 0.0, a.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_fans_into_triangles() {
        let mut m = MeshBuilder::new();
        m.polygon(&[Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y]);
        assert_eq!(m.vertex_count(), 6);
    }

    #[test]
    fn segments_can_change() {
        let s = Shape::Sphere { radius: 1.0, segments: 8 }.with_segments(16);
        assert_eq!(s.segments(), Some(16));
        assert!(s.build().vertex_count() > Shape::Sphere { radius: 1.0, segments: 8 }.build().vertex_count());
    }

    #[test]
    fn flatten_makes_faces_share_normal() {
        let mut m = MeshBuilder::sphere(1.0, 6);
        m.flatten();
        for tri in m.data().chunks_exact(18) {
            assert_eq!(tri[3..6], tri[9..12]);
            assert_eq!(tri[3..6], tri[15..18]);
        }
    }

    #[test]
    fn primitives_have_expected_counts() {
        assert_eq!(MeshBuilder::cube(2.0).vertex_count(), 36);
        assert_eq!(MeshBuilder::prism(5, 1.0, 1.0).vertex_count(), 5 * 12);
        assert_eq!(MeshBuilder::pyramid(4, 1.0, 1.0).vertex_count(), 4 * 6);
        assert_eq!(MeshBuilder::cone(1.0, 1.0, 8).vertex_count(), 8 * 6);
        assert_eq!(MeshBuilder::regular_polygon(6, 1.0).vertex_count(), 4 * 3);
    }

    #[test]
    fn solid_normals_point_outward() {
        let m = MeshBuilder::pyramid(5, 1.0, 2.0);
        for v in m.data().chunks_exact(6) {
            let (p, n) = (Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5]));
            assert!(n.dot(p) > -1e-4 || p.y.abs() > 0.99, "{p:?} {n:?}");
        }
    }
}
