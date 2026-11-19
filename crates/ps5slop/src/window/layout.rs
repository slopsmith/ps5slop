//! Window geometry, kept apart from winit and Metal. From 86slopper.

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Where a `picture` is drawn in a `drawable` (both in pixels): centred with
/// its aspect ratio kept, between black bars. The flag is true for a
/// whole-number scale, drawn nearest-neighbour so guest pixels stay sharp;
/// a scale within 1% of one snaps to it. Other scales filter linearly.
pub fn draw_rect(drawable: (u32, u32), picture: (u32, u32)) -> (Rect, bool) {
    let (dw, dh) = (drawable.0 as f64, drawable.1 as f64);
    let (pw, ph) = (picture.0.max(1) as f64, picture.1.max(1) as f64);
    let fit = (dw / pw).min(dh / ph);
    let whole = fit.floor();
    let nearest = whole >= 1.0 && whole >= fit * 0.99;
    let scale = if nearest { whole } else { fit };
    let (w, h) = ((pw * scale).round(), (ph * scale).round());
    let rect = Rect {
        x: ((dw - w) / 2.0).floor(),
        y: ((dh - h) / 2.0).floor(),
        w,
        h,
    };
    (rect, nearest)
}

/// The window's inner size in points for a `picture`: the picture's own
/// size, scaled down to fit a `screen` of that many points if it is larger.
pub fn window_size_for(picture: (u32, u32), screen: Option<(f64, f64)>) -> (f64, f64) {
    let (w, h) = (picture.0.max(1) as f64, picture.1.max(1) as f64);
    let scale = screen.map_or(1.0, |(sw, sh)| (sw * 0.95 / w).min(sh * 0.9 / h).min(1.0));
    ((w * scale).floor(), (h * scale).floor())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_twice_its_size_is_sharp() {
        let (r, nearest) = draw_rect((1600, 1200), (800, 600));
        assert_eq!(
            r,
            Rect {
                x: 0.0,
                y: 0.0,
                w: 1600.0,
                h: 1200.0
            }
        );
        assert!(nearest);
    }

    #[test]
    fn a_wide_window_has_bars_left_and_right() {
        let (r, nearest) = draw_rect((1200, 600), (800, 600));
        assert_eq!(
            r,
            Rect {
                x: 200.0,
                y: 0.0,
                w: 800.0,
                h: 600.0
            }
        );
        assert!(nearest);
    }

    #[test]
    fn a_fractional_scale_fills_the_space_with_linear_filtering() {
        let (r, nearest) = draw_rect((1920, 1080), (800, 600));
        assert_eq!(
            r,
            Rect {
                x: 240.0,
                y: 0.0,
                w: 1440.0,
                h: 1080.0
            }
        );
        assert!(!nearest);
        let (r, nearest) = draw_rect((700, 500), (800, 600));
        assert!(!nearest && r.w <= 700.0 && r.h <= 500.0);
    }

    #[test]
    fn windows_fit_the_picture_or_the_screen() {
        assert_eq!(
            window_size_for((800, 600), Some((1512.0, 982.0))),
            (800.0, 600.0)
        );
        assert_eq!(
            window_size_for((1440, 900), Some((1512.0, 982.0))),
            (1414.0, 883.0)
        );
        assert_eq!(window_size_for((1920, 1080), None), (1920.0, 1080.0));
    }
}
