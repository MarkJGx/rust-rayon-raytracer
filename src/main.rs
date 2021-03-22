#[macro_use]
extern crate lazy_static;
extern crate glam;
extern crate image as im;
extern crate piston_window;

use glam::*;
use piston_window::clear;
use piston_window::image;
use piston_window::math::Matrix2d;
use piston_window::G2dTexture;
use piston_window::OpenGL;
use piston_window::PistonWindow;
use piston_window::RenderEvent;
use piston_window::Texture;
use piston_window::TextureContext;
use piston_window::TextureSettings;
use piston_window::Transformed;
use piston_window::WindowSettings;
use std::time::{SystemTime, UNIX_EPOCH};

use fastrand::*;
use rand::rngs::mock::StepRng;
use rand::Rng;

type Color = Vec3A;

lazy_static! {
    static ref ZERO: Vec3A = Vec3A::new(0.0, 0.0, 0.0);
    static ref UP: Vec3A = Vec3A::new(0.0, 0.0, 1.0);
    static ref DOWN: Vec3A = Vec3A::new(0.0, 0.0, -1.0);
    static ref ONE: Vec3A = Vec3A::new(1.0, 1.0, 1.0);
}

#[inline]
fn random() -> f32 {
    return fastrand::f32();
}

#[inline]
fn random_minmax(min: f32, max: f32) -> f32 {
    let m: f32 = random();
    return (m * (max - min)) + min;
}

fn random_in_unit_sphere() -> Vec3A {
    let min = -1.0;
    let max = 1.0;

    let mut point = Vec3A::new(0.0, 0.0, 0.0);

    while (true) {
        point = Vec3A::new(
            random_minmax(min, max),
            random_minmax(min, max),
            random_minmax(min, max),
        );
        if point.length_squared() >= 1.0 {
            continue;
        }
        break;
    }

    return point;
}

struct Camera {
    origin: Vec3A,
    lower_left_corner: Vec3A,
    horizontal: Vec3A,
    vertical: Vec3A,
}

impl Camera {
    fn new(aspect_ratio: f32) -> Camera {
        let viewport_width = 2.0;
        let viewport_height = viewport_width / aspect_ratio;
        let focal_length = 1.0;
        let origin: Vec3A = Vec3A::new(0.0, 0.0, 0.0);
        let horizontal: Vec3A = Vec3A::new(viewport_width, 0.0, 0.0);
        let vertical: Vec3A = Vec3A::new(0.0, viewport_height, 0.0);
        let lower_left_corner: Vec3A = origin
            - Vec3A::new(0.0, viewport_width, 0.0) / 2.0
            - Vec3A::new(viewport_height, 0.0, 0.0) / 2.0
            - Vec3A::new(0.0, 0.0, focal_length);

        return Camera {
            origin: origin,
            lower_left_corner: lower_left_corner,
            horizontal: horizontal,
            vertical: vertical,
        };
    }

    #[inline]
    fn get_ray(&self, u: f32, v: f32) -> Ray {
        let ray = Ray {
            origin: self.origin,
            direction: self.lower_left_corner + (u * self.horizontal) + (v * self.vertical)
                - self.origin,
        };

        return ray;
    }
}

struct Ray {
    origin: Vec3A,
    direction: Vec3A,
}

impl Ray {
    fn at(&self, t: f32) -> Vec3A {
        return self.origin + (self.direction * t);
    }
}

#[derive(Copy, Clone)]
struct HitRecord {
    point: Vec3A,
    normal: Vec3A,
    hit_dist: f32,
    front_face: bool,
}

impl Default for HitRecord {
    fn default() -> HitRecord {
        HitRecord {
            point: *ZERO,
            normal: *ZERO,
            hit_dist: 0.0,
            front_face: true,
        }
    }
}

impl HitRecord {
    #[inline]
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vec3A) {
        self.front_face = ray.direction.dot(*outward_normal) < 0.0;
        self.normal = match self.front_face {
            true => *outward_normal,
            false => -*outward_normal,
        }
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit_record: &mut HitRecord) -> bool;
}

struct Sphere {
    radius: f32,
    origin: Vec3A,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit_record: &mut HitRecord) -> bool {
        let oc: Vec3A = ray.origin - self.origin;
        let a = ray.direction.length_squared();

        let half_b = oc.dot(ray.direction);
        let c = oc.length_squared() - (self.radius * self.radius);

        let d = (half_b * half_b) - (a * c);
        if d < 0.0 {
            return false;
        }

        let sqrtd = d.sqrt();

        // Find the nearest root that lies in the acceptable range.
        // @MarkJGx: We do this because we don't want to return the back of the sphere.
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return false;
            }
        }

        hit_record.hit_dist = root;
        hit_record.point = ray.at(hit_record.hit_dist);
        let outward_normal: Vec3A =
            (hit_record.point - self.origin) / self.radius.max(f32::EPSILON);
        hit_record.set_face_normal(ray, &outward_normal);
        return true;
    }
}

pub struct HittableScene {
    hittables_list: Vec<Box<dyn Hittable>>,
}

impl Hittable for HittableScene {
    #[inline]
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit_record: &mut HitRecord) -> bool {
        let mut temp_hit_record: HitRecord = { Default::default() };
        let mut hit_anything: bool = false;
        let mut closest_dist_so_far: f32 = t_max;

        for hittable in self.hittables_list.iter() {
            if hittable.hit(ray, t_min, closest_dist_so_far, &mut temp_hit_record) {
                hit_anything = true;
                closest_dist_so_far = temp_hit_record.hit_dist;

                hit_record.normal = temp_hit_record.normal;
                hit_record.point = temp_hit_record.point;
                hit_record.hit_dist = temp_hit_record.hit_dist;
                hit_record.front_face = temp_hit_record.front_face;
            }
        }

        return hit_anything;
    }
}

#[inline]
fn ray_color(ray: &Ray, scene: &HittableScene, depth: &i32) -> Color {
    let mut hit_record: HitRecord = { Default::default() };

    // If we've exceeded the ray bounce limit, no more light is gathered
    if *depth <= 0 {
        return *ZERO;
    }

    if scene.hit(ray, 0.0, f32::INFINITY, &mut hit_record) {
        let target = hit_record.point + hit_record.normal + random_in_unit_sphere();

        return 0.5
            * ray_color(
                &Ray {
                    origin: hit_record.point,
                    direction: target - hit_record.point,
                },
                scene,
                &(*depth - 1),
            );
    }

    // background
    let time = 0.5 * (ray.direction.normalize().y + 1.0);
    return (1.0 - time) * Color::new(1.0, 1.0, 1.0) + (time * Color::new(0.5, 0.7, 1.0));
}

fn get_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

fn main() {
    let aspect_ratio = 9.0 / 16.0;
    let image_width: i32 = (200.0) as i32;
    let image_height: i32 = ((image_width as f32) * aspect_ratio) as i32;

    let camera: Camera = Camera::new(aspect_ratio);

    let window_scale = 4.0;

    let mut window: PistonWindow = WindowSettings::new(
        "Raytracer",
        (
            (image_width as f32 * window_scale) as u32,
            (image_height as f32 * window_scale) as u32,
        ),
    )
    .exit_on_esc(true)
    .vsync(false)
    .resizable(false)
    .graphics_api(OpenGL::V3_2)
    .build()
    .unwrap();

    println!(
        "Created canvas width {}, height {}",
        image_width, image_height
    );
    let mut canvas = im::ImageBuffer::new(image_width as u32, image_height as u32);
    let mut texture_context = TextureContext {
        factory: window.factory.clone(),
        encoder: window.factory.create_command_buffer().into(),
    };

    let mut texture: G2dTexture =
        Texture::from_image(&mut texture_context, &canvas, &TextureSettings::new()).unwrap();

    let mut time: u128 = get_epoch_ms();
    let mut delta: f32 = 0.0;

    let mut tick: i32 = 0;

    let mut scene: HittableScene = HittableScene {
        hittables_list: Vec::new(),
    };
    scene.hittables_list.push(Box::new(Sphere {
        origin: Vec3A::new(0.0, 0.0, -1.0),
        radius: 0.5,
    }));

    scene.hittables_list.push(Box::new(Sphere {
        origin: Vec3A::new(0.0, -100.5, -1.0),
        radius: 100.0,
    }));

    let max_depth: u32 = 2;

    let samples_per_pixel: i32 = 100;

    while let Some(event) = window.next() {
        if let Some(_) = event.render_args() {
            texture.update(&mut texture_context, &canvas).unwrap();
            window.draw_2d(&event, |context, graphics, device| {
                // Update texture before rendering.
                texture_context.encoder.flush(device);

                // clear([0.0; 4], graphics);
                let scaled_context = context.scale(window_scale as f64, window_scale as f64);
                image(&texture, scaled_context.transform, graphics);
            });

            let mut write_color = |position: &UVec2, color: &Color, samples_per_pixel: &i32| {
                let mut scaled_color = *color;

                let scale = 1.0 / *samples_per_pixel as f32;
                scaled_color *= scale;

                canvas.put_pixel(
                    position.x,
                    (image_height as u32 - 1) - position.y,
                    im::Rgba([
                        (scaled_color.x as f32 * 255.999) as u8,
                        (scaled_color.y as f32 * 255.999) as u8,
                        (scaled_color.z as f32 * 255.999) as u8,
                        255,
                    ]),
                )
            };

            for y in (0..image_height).rev() {
                for x in 0..image_width {
                    let mut color: Color = Color::new(0.0, 0.0, 0.0);

                    for sample in 0..samples_per_pixel {
                        let mut rng_x = 0.0;
                        let mut rng_y = 0.0;

                        if (samples_per_pixel > 1) {
                            rng_x = random();
                            rng_y = random();
                        }

                        let u: f32 = ((x as f32) + rng_x) / (image_height as f32 - 1.0);
                        let v: f32 = ((y as f32) + rng_y) / (image_width as f32 - 1.0);

                        let ray: Ray = camera.get_ray(u, v);
                        color += ray_color(&ray, &scene, &(max_depth as i32));
                    }

                    write_color(&UVec2::new(x as u32, y as u32), &color, &samples_per_pixel);
                }
            }

            let new_time = get_epoch_ms();
            delta = (new_time - time) as f32;
            time = new_time;
            if tick % 5 == 0 {
                println!(
                    "Framerate: {}, Frametime {}ms",
                    (1000.0 / delta) as i32,
                    delta
                );
            }
            tick += 1;
        }
    }
}
