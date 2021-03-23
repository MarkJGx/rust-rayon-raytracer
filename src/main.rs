// Following https://raytracing.github.io/books/RayTracingInOneWeekend.html

#[macro_use]
extern crate lazy_static;
extern crate glam;
extern crate image as im;
extern crate piston_window;
extern crate rayon;

use dyn_clone::DynClone;
use fastrand::*;
use glam::*;
use piston_window::clear;
use piston_window::image;
use piston_window::math::Matrix2d;
use piston_window::*;
use rayon::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

type Color = Vec3A;

fn get_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

lazy_static! {
    static ref ZERO: Vec3A = Vec3A::new(0.0, 0.0, 0.0);
    static ref UP: Vec3A = Vec3A::new(0.0, 0.0, 1.0);
    static ref DOWN: Vec3A = Vec3A::new(0.0, 0.0, -1.0);
    static ref ONE: Vec3A = Vec3A::new(1.0, 1.0, 1.0);
}

#[inline]
fn is_nearly_zero(vector: &Vec3A) -> bool {
    let s: f32 = 1e-8;
    return (vector.x < s) && (vector.y < s) && (vector.z < s);
}

#[inline]
fn reflectance(cosine: &f32, ref_idx: &f32) -> f32 {
    // Use Schlick's approximation for reflectance.
    let mut r0 = (1.0 - *ref_idx) / (1.0 + *ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * (1.0 - *cosine).powf(5.0);
}

#[inline]
fn refract(uv: &Vec3A, normal: &Vec3A, etai_over_etat: &f32) -> Vec3A {
    // no idea how this works.
    let cos_theta = f32::min(((-*uv).dot(*normal)), 1.0);
    let r_out_perp: Vec3A = *etai_over_etat * (*uv + (cos_theta * *normal));
    let r_out_parallel = -f32::sqrt(f32::abs(1.0 - r_out_perp.length_squared())) * *normal;
    return r_out_perp + r_out_parallel;
}

#[inline]
fn reflect(v: &Vec3A, normal: &Vec3A) -> Vec3A {
    return *v - (2.0 * v.dot(*normal)) * *normal;
}

#[inline]
fn random() -> f32 {
    // actually expensive
    return fastrand::f32();
}

#[inline]
fn random_minmax(min: f32, max: f32) -> f32 {
    let m: f32 = random();
    return (m * (max - min)) + min;
}

// mathworld.wolfram.com/SpherePointPicking.html
// https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
#[inline]
fn random_unit_in_sphere() -> Vec3A {
    let u = random();
    let v = random();
    let theta = u * 2.0 * std::f32::consts::PI;
    let phi = f32::acos(2.0 * v - 1.0);
    let r = random().cbrt();
    let sin_theta = f32::sin(theta);
    let cos_theta = f32::cos(theta);
    let sin_phi = f32::sin(phi);
    let cos_hi = f32::cos(phi);
    let x = r * sin_phi * cos_theta;
    let y = r * sin_phi * sin_theta;
    let z = r * cos_hi;
    return Vec3A::new(x, y, z);
}

#[inline]
fn random_unit_vector() -> Vec3A {
    return random_unit_in_sphere().normalize();
}

#[inline]
fn random_in_hemisphere(normal: Vec3A) -> Vec3A {
    let random_unit = random_unit_in_sphere();
    if random_unit.dot(normal) > 0.0 {
        return random_unit;
    } else {
        return -random_unit;
    }
}

struct Camera {
    origin: Vec3A,
    lower_left_corner: Vec3A,
    horizontal: Vec3A,
    vertical: Vec3A,
}

impl Camera {
    fn new(
        look_from: Vec3A,
        look_at: Vec3A,
        up: Vec3A,
        vertical_fov: f32,
        aspect_ratio: f32,
    ) -> Camera {
        let theta: f32 = vertical_fov.to_radians();
        let h = f32::tan(theta / 2.0);

        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let direciton_w = (look_from - look_at).normalize();
        let u = up.cross(direciton_w).normalize();
        let v = direciton_w.cross(u);

        let focal_length = 1.0;
        let origin: Vec3A = look_from;
        let horizontal: Vec3A = viewport_width * u;
        let vertical: Vec3A = viewport_height * v;
        let lower_left_corner: Vec3A = origin - (horizontal / 2.0) - (vertical / 2.0) - direciton_w;

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

#[derive(Default)]
struct Ray {
    origin: Vec3A,
    direction: Vec3A,
}

impl Ray {
    #[inline]
    fn at(&self, t: f32) -> Vec3A {
        return self.origin + (self.direction * t);
    }

    #[inline]
    fn new(origin: Vec3A, direction: Vec3A) -> Ray {
        let new: Ray = Ray {
            origin: origin,
            direction: direction,
        };

        return new;
    }
}

#[derive(Clone)]
struct HitRecord {
    point: Vec3A,
    normal: Vec3A,
    hit_dist: f32,
    front_face: bool,
    material: Box<MaterialTrait + Send + Sync>,
}

impl Default for HitRecord {
    fn default() -> HitRecord {
        let material = Lambertian {
            albedo: Vec3A::new(0.5, 0.5, 0.5),
        };

        HitRecord {
            point: *ZERO,
            normal: *ZERO,
            hit_dist: 0.0,
            front_face: true,
            material: Box::new(material),
        }
    }
}

trait MaterialTrait: DynClone {
    fn scatter(
        &self,
        ray: &Ray,
        hit_record: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool;
}
dyn_clone::clone_trait_object!(MaterialTrait);

#[derive(Clone)]
struct Lambertian {
    albedo: Color,
}

impl MaterialTrait for Lambertian {
    #[inline]
    fn scatter(
        &self,
        ray: &Ray,
        hit_record: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        let mut scatter_direction = hit_record.normal + random_unit_vector();

        if is_nearly_zero(&scatter_direction) {
            scatter_direction = hit_record.normal;
        }

        *attenuation = self.albedo;

        scattered.origin = hit_record.point;
        scattered.direction = scatter_direction;

        return true;
    }
}

#[derive(Clone)]
struct Metal {
    albedo: Color,
    fuzz: f32,
}

impl MaterialTrait for Metal {
    #[inline]
    fn scatter(
        &self,
        ray: &Ray,
        hit_record: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        let reflected: Vec3A = reflect(&ray.direction.normalize(), &hit_record.normal);
        scattered.origin = hit_record.point;

        let mut computed_fuzz: Vec3A = *ZERO;
        if self.fuzz > 0.0 {
            computed_fuzz = self.fuzz * random_unit_in_sphere();
        }
        scattered.direction = reflected + (computed_fuzz);

        *attenuation = self.albedo;

        return scattered.direction.dot(hit_record.normal) > 0.0;
    }
}

#[derive(Clone)]
struct Dielectric {
    ir: f32,
}

impl MaterialTrait for Dielectric {
    #[inline]
    fn scatter(
        &self,
        ray: &Ray,
        hit_record: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        attenuation.x = 1.0;
        attenuation.y = 1.0;
        attenuation.z = 1.0;

        let refraction_ratio = if hit_record.front_face {
            1.0 / self.ir
        } else {
            self.ir
        };

        let unit_direction: Vec3A = ray.direction.normalize();
        let cos_theta: f32 = f32::min((-unit_direction).dot(hit_record.normal), 1.0);

        let sin_theta = f32::sqrt(1.0 - (cos_theta * cos_theta));

        let cannot_refract: bool = refraction_ratio * sin_theta > 1.0;
        let mut direction: Vec3A;
        if cannot_refract || reflectance(&cos_theta, &refraction_ratio) > random() {
            direction = reflect(&unit_direction, &hit_record.normal);
        } else {
            direction = refract(&unit_direction, &hit_record.normal, &refraction_ratio);
        }

        scattered.origin = hit_record.point;
        scattered.direction = direction;
        return true;
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
    material: Box<MaterialTrait + Send + Sync>,
}

impl Hittable for Sphere {
    #[inline]
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
        hit_record.material = dyn_clone::clone_box(&*self.material);
        return true;
    }
}

pub struct HittableScene {
    hittables_list: Vec<Box<dyn Hittable + Send + Sync>>,
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
                hit_record.material = dyn_clone::clone_box(&*temp_hit_record.material);
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

    if scene.hit(ray, 0.001, f32::INFINITY, &mut hit_record) {
        let mut scattered: Ray = Default::default();
        let mut attenuation: Color = Vec3A::new(0.0, 0.0, 0.0);
        if hit_record
            .material
            .scatter(&ray, &hit_record, &mut attenuation, &mut scattered)
        {
            return attenuation * ray_color(&scattered, &scene, &(*depth - 1));
        }
    }

    // background
    let time = 0.5 * (ray.direction.normalize().y + 1.0);
    return (1.0 - time) * Color::new(1.0, 1.0, 1.0) + (time * Color::new(0.5, 0.7, 1.0));
}

fn main() {
    let aspect_ratio = 9.0 / 16.0;
    let image_width: u32 = 256 * 2;
    let image_height: u32 = ((image_width as f32) * aspect_ratio) as u32;

    let max_depth: u32 = 4;
    let samples_per_pixel: i32 = 5;

    let mut camera: Camera = Camera::new(
        Vec3A::new(0.0, 0.0, 1.0),
        Vec3A::new(0.0, 0.0, -1.0),
        Vec3A::new(0.0, 1.0, 0.0),
        90.0,
        aspect_ratio,
    );
    let window_scale = 2.0;

    let mut window: PistonWindow = WindowSettings::new(
        "Rust Rayon Raytracer",
        (
            (image_width as f32 * window_scale) as u32,
            (image_height as f32 * window_scale) as u32,
        ),
    )
    .exit_on_esc(true)
    .vsync(true)
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

    let mut scene: HittableScene = HittableScene {
        hittables_list: Vec::new(),
    };

    // let r = f32::cos(std::f32::consts::PI / 4.0);
    // scene.hittables_list.push(Box::new(Sphere {
    //     origin: Vec3A::new(-r, 0.0, -1.0),
    //     radius: r,
    //     material: Box::new(Lambertian {
    //         albedo: Color::new(0.0, 0.0, 1.0),
    //     }),
    // }));

    // scene.hittables_list.push(Box::new(Sphere {
    //     origin: Vec3A::new(r, 0.0, -1.0),
    //     radius: r,
    //     material: Box::new(Lambertian {
    //         albedo: Color::new(1.0, 0.0, 0.0),
    //     }),
    // }));

    //Ground
    scene.hittables_list.push(Box::new(Sphere {
        origin: Vec3A::new(0.0, -100.5, -1.0),
        radius: 100.0,
        material: Box::new(Lambertian {
            albedo: Color::new(0.8, 0.8, 0.0),
        }),
    }));

    scene.hittables_list.push(Box::new(Sphere {
        origin: Vec3A::new(0.0, 0.0, -1.0),
        radius: 0.5,
        material: Box::new(Lambertian {
            albedo: Color::new(0.1, 0.2, 0.5),
        }),
    }));

    scene.hittables_list.push(Box::new(Sphere {
        origin: Vec3A::new(-1.0, 0.0, -1.0),
        radius: -0.4,
        material: Box::new(Dielectric { ir: 1.5 }),
    }));

    scene.hittables_list.push(Box::new(Sphere {
        origin: Vec3A::new(1.0, 0.0, -1.0),
        radius: 0.5,
        material: Box::new(Metal {
            albedo: Color::new(0.8, 0.6, 0.2),
            fuzz: 0.0,
        }),
    }));

    let mut time: u128 = get_epoch_ms();
    let mut delta: f32 = 0.0;

    let mut tick: i32 = 0;

    let pixel_amount = image_width * image_height;
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

                // Divide the color by the number of samples and
                let scale = 1.0 / *samples_per_pixel as f32;
                scaled_color *= scale;

                // Gamma-correct for gamma=2.0
                scaled_color.x = scaled_color.x.sqrt();
                scaled_color.y = scaled_color.y.sqrt();
                scaled_color.z = scaled_color.z.sqrt();

                scaled_color *= 255.999;

                canvas.put_pixel(
                    position.x,
                    image_height - 1 - position.y,
                    im::Rgba([
                        scaled_color.x as u8,
                        scaled_color.y as u8,
                        scaled_color.z as u8,
                        255,
                    ]),
                )
            };

            let pixels: Vec<Vec3A> = (0..pixel_amount)
                .into_par_iter()
                .map(|pixel_index: u32| {
                    let x: u32 = pixel_index % image_width;
                    let y: u32 = (pixel_index as f32 / image_width as f32) as u32;

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

                    return color;
                })
                .collect();

            for (index, color) in pixels.iter().enumerate() {
                let x: u32 = index as u32 % image_width;
                let y: u32 = (index as f32 / image_width as f32) as u32;
                let mut new_color = *color;
                write_color(&UVec2::new(x, y), &new_color, &samples_per_pixel);
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
