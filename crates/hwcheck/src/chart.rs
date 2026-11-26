//! The instruction status page: every gfx1013 instruction form, grouped by
//! encoding, coloured by whether the translator takes it and whether real
//! hardware agrees with it.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gfx10::opinfo::{Fmt, OpInfo};

use crate::coverage::{Coverage, scan};

/// Where one instruction form stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    /// Translated, and every hardware test of it agrees.
    Agrees,
    /// Translated, and some hardware test of it differs in any way.
    Fails,
    /// Translated, not tested on hardware yet.
    Untested,
    /// Translated; the reference GPU lacks the form or runs it differently
    /// from gfx1013, so it cannot be tested there.
    Untestable,
    /// An instruction form the translator refuses.
    Missing,
}

impl Status {
    fn colour(self) -> &'static str {
        match self {
            Status::Agrees => "var(--agrees)",
            Status::Fails => "var(--fails)",
            Status::Untested => "var(--untested)",
            Status::Untestable => "var(--untestable)",
            Status::Missing => "var(--missing)",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Status::Agrees => "matches the reference GPU",
            Status::Fails => "differs from the reference GPU",
            Status::Untested => "translated, not tested yet",
            Status::Untestable => "translated, runs differently on the reference GPU's ISA",
            Status::Missing => "not translated",
        }
    }
}

/// One line of a results file: opcode, variant, outcome, details.
struct Outcome {
    variant: String,
    outcome: String,
    details: String,
}

/// Results files as `ps5slop-hwcheck run --results` writes them, by opcode.
fn parse_results(text: &str) -> BTreeMap<String, Vec<Outcome>> {
    let mut out: BTreeMap<String, Vec<Outcome>> = BTreeMap::new();
    for line in text.lines() {
        let mut f = line.splitn(4, '\t');
        let (Some(op), Some(variant), Some(outcome)) = (f.next(), f.next(), f.next()) else {
            continue;
        };
        out.entry(op.to_string()).or_default().push(Outcome {
            variant: variant.to_string(),
            outcome: outcome.to_string(),
            details: f.next().unwrap_or_default().to_string(),
        });
    }
    out
}

/// One cell of the chart.
pub struct Form {
    pub group: &'static str,
    pub name: String,
    pub status: Status,
    pub note: String,
}

/// Which encoding of an opcode a cell stands for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Base,
    Vop3,
    Dpp,
    Sdwa,
}

impl Shape {
    fn suffix(self) -> &'static str {
        match self {
            Shape::Base => "",
            Shape::Vop3 => "_e64",
            Shape::Dpp => "_dpp",
            Shape::Sdwa => "_sdwa",
        }
    }

    /// Whether a result's variant tests this shape.
    fn picks(self, variant: &str, has_e32: bool) -> bool {
        match self {
            Shape::Base if has_e32 => !["e64", "dpp", "sdwa"]
                .iter()
                .any(|p| variant.starts_with(p)),
            Shape::Base => true,
            Shape::Vop3 => variant.starts_with("e64"),
            Shape::Dpp => variant.starts_with("dpp"),
            Shape::Sdwa => variant.starts_with("sdwa"),
        }
    }
}

/// The encodings `op` has: its own, and VOP3, DPP and SDWA where LLVM takes them.
fn shapes(op: &OpInfo) -> Vec<Shape> {
    use crate::coverage::{Promotion, has};
    let mut out = vec![Shape::Base];
    for (p, shape) in [
        (Promotion::Vop3, Shape::Vop3),
        (Promotion::Dpp, Shape::Dpp),
        (Promotion::Sdwa, Shape::Sdwa),
    ] {
        if has(op, p) {
            out.push(shape);
        }
    }
    out
}

fn group(op: &OpInfo, shape: Shape) -> &'static str {
    match (op.fmt, shape) {
        (Fmt::Vop1, Shape::Vop3) => "VOP1 in VOP3",
        (Fmt::Vop2, Shape::Vop3) => "VOP2 in VOP3",
        (Fmt::Vopc, Shape::Vop3) => "VOPC in VOP3",
        (Fmt::Vop1, Shape::Dpp) => "VOP1 DPP",
        (Fmt::Vop2, Shape::Dpp) => "VOP2 DPP",
        (Fmt::Vop1, Shape::Sdwa) => "VOP1 SDWA",
        (Fmt::Vop2, Shape::Sdwa) => "VOP2 SDWA",
        (Fmt::Vopc, Shape::Sdwa) => "VOPC SDWA",
        (Fmt::Sop1, _) => "SOP1",
        (Fmt::Sop2, _) => "SOP2",
        (Fmt::Sopk, _) => "SOPK",
        (Fmt::Sopp, _) => "SOPP",
        (Fmt::Sopc, _) => "SOPC",
        (Fmt::Smem, _) => "SMEM",
        (Fmt::Ds, _) => "DS",
        (Fmt::Mtbuf, _) => "MTBUF",
        (Fmt::Mubuf, _) => "MUBUF",
        (Fmt::Mimg, _) => "MIMG",
        (Fmt::Exp, _) => "EXP",
        (Fmt::Flat, _) => "FLAT",
        (Fmt::Global, _) => "GLOBAL",
        (Fmt::Scratch, _) => "SCRATCH",
        (Fmt::Vintrp, _) => "VINTRP",
        (Fmt::Vop1, _) => "VOP1",
        (Fmt::Vop2, _) => "VOP2",
        (Fmt::Vopc, _) => "VOPC",
        (Fmt::Vop3, _) => "VOP3",
        (Fmt::Vop3p, _) => "VOP3P",
    }
}

fn judge(outcomes: &[&Outcome]) -> (Status, String) {
    let ran: Vec<&&Outcome> = outcomes
        .iter()
        .filter(|o| matches!(o.outcome.as_str(), "agree" | "differ" | "hung"))
        .collect();
    if ran.is_empty() {
        return match outcomes.iter().find(|o| o.outcome == "untranslated") {
            Some(o) => (Status::Missing, format!("{}: {}", o.variant, o.details)),
            None => (Status::Untested, String::new()),
        };
    }
    let failing: Vec<String> = ran
        .iter()
        .filter(|o| o.outcome != "agree")
        .map(|o| format!("{}: {} {}", o.variant, o.outcome, o.details))
        .collect();
    if failing.is_empty() {
        (Status::Agrees, format!("{} tests agree", ran.len()))
    } else {
        (Status::Fails, failing.join("\n"))
    }
}

/// Every instruction form with its status, from the translator, whether
/// `cpu` can run it, and `results` from that GPU.
pub fn forms(results: &str, cpu: &str) -> Vec<Form> {
    let results = parse_results(results);
    let mut out = Vec::new();
    for (op, coverage) in scan() {
        let untestable = crate::coverage::untestable(op, cpu);
        let has_e32 = matches!(op.fmt, Fmt::Vop1 | Fmt::Vop2 | Fmt::Vopc);
        let outcomes = results.get(op.name).map(Vec::as_slice).unwrap_or_default();
        for shape in shapes(op) {
            let (status, note) = match &coverage {
                Coverage::Untranslated(why) => (Status::Missing, why.clone()),
                Coverage::Unknown if untestable.is_some() => {
                    (Status::Untestable, untestable.clone().unwrap_or_default())
                }
                Coverage::Unknown => (Status::Untested, "not scanned".into()),
                Coverage::Translated if untestable.is_some() => {
                    (Status::Untestable, untestable.clone().unwrap_or_default())
                }
                Coverage::Translated => {
                    let mine: Vec<&Outcome> = outcomes
                        .iter()
                        .filter(|o| shape.picks(&o.variant, has_e32))
                        .collect();
                    judge(&mine)
                }
            };
            out.push(Form {
                group: group(op, shape),
                name: format!("{}{}", op.name, shape.suffix()),
                status,
                note,
            });
        }
    }
    out
}

struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

/// Squarified treemap (Bruls, Huizing and van Wijk): lays `sizes` (largest
/// first) out in `area`, keeping each rectangle close to square.
fn squarify(sizes: &[f64], area: Rect) -> Vec<Rect> {
    let total: f64 = sizes.iter().sum();
    let scale = area.w * area.h / total;
    let mut out = Vec::new();
    let mut rest = area;
    let mut i = 0;
    while i < sizes.len() {
        let side = rest.w.min(rest.h);
        let worst = |row: &[f64]| {
            let s: f64 = row.iter().sum::<f64>() * scale;
            let (max, min) = row.iter().fold((0f64, f64::MAX), |(a, b), &r| {
                (a.max(r * scale), b.min(r * scale))
            });
            (side * side * max / (s * s)).max(s * s / (side * side * min))
        };
        let mut end = i + 1;
        while end < sizes.len() && worst(&sizes[i..=end]) <= worst(&sizes[i..end]) {
            end += 1;
        }
        let row = &sizes[i..end];
        let row_area: f64 = row.iter().sum::<f64>() * scale;
        let thick = row_area / side;
        let mut at = 0.0;
        for &r in row {
            let len = r * scale / thick;
            if rest.w >= rest.h {
                out.push(Rect {
                    x: rest.x,
                    y: rest.y + at,
                    w: thick,
                    h: len,
                });
            } else {
                out.push(Rect {
                    x: rest.x + at,
                    y: rest.y,
                    w: len,
                    h: thick,
                });
            }
            at += len;
        }
        if rest.w >= rest.h {
            rest = Rect {
                x: rest.x + thick,
                w: rest.w - thick,
                ..rest
            };
        } else {
            rest = Rect {
                y: rest.y + thick,
                h: rest.h - thick,
                ..rest
            };
        }
        i = end;
    }
    out
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The status page as one self-contained HTML document. `source` says
/// which hardware run and commit the results come from.
pub fn html(forms: &[Form], source: &str) -> String {
    let mut groups: BTreeMap<&str, Vec<&Form>> = BTreeMap::new();
    for f in forms {
        groups.entry(f.group).or_default().push(f);
    }
    let mut order: Vec<(&str, Vec<&Form>)> = groups.into_iter().collect();
    order.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
    for (_, v) in &mut order {
        v.sort_by(|a, b| a.status.cmp(&b.status).then(a.name.cmp(&b.name)));
    }
    let (w, h) = (1200.0, 700.0);
    let sizes: Vec<f64> = order.iter().map(|(_, v)| v.len() as f64).collect();
    let rects = squarify(
        &sizes,
        Rect {
            x: 0.0,
            y: 0.0,
            w,
            h,
        },
    );

    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg viewBox="0 0 {w} {h}" role="img" aria-label="Instruction status treemap">"#
    );
    for ((name, cells), r) in order.iter().zip(&rects) {
        let pad = 2.0;
        let header = 16.0;
        let (iw, ih) = (r.w - 2.0 * pad, r.h - 2.0 * pad - header);
        let n = cells.len() as f64;
        let cols = ((n * iw / ih.max(1.0)).sqrt().ceil()).max(1.0);
        let rows = (n / cols).ceil();
        let (cw, ch) = (iw / cols, ih / rows);
        let _ = write!(
            svg,
            r#"<g><rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" class="group"/>"#,
            r.x, r.y, r.w, r.h
        );
        for (k, f) in cells.iter().enumerate() {
            let (cx, cy) = ((k as f64 % cols).floor(), (k as f64 / cols).floor());
            let _ = write!(
                svg,
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" class="cell"><title>{}: {}{}{}</title></rect>"#,
                r.x + pad + cx * cw,
                r.y + pad + header + cy * ch,
                (cw - 0.6).max(0.5),
                (ch - 0.6).max(0.5),
                f.status.colour(),
                escape(&f.name),
                f.status.label(),
                if f.note.is_empty() { "" } else { "\n" },
                escape(&f.note)
            );
        }
        let done = cells.iter().filter(|f| f.status != Status::Missing).count();
        let _ = write!(
            svg,
            r#"<text x="{:.1}" y="{:.1}" class="label">{} {}/{}</text></g>"#,
            r.x + 4.0,
            r.y + 13.0,
            name,
            done,
            cells.len()
        );
    }
    svg.push_str("</svg>");

    let count = |s: Status| forms.iter().filter(|f| f.status == s).count();
    let total = forms.len();
    let mut legend = String::new();
    for s in [
        Status::Agrees,
        Status::Fails,
        Status::Untested,
        Status::Untestable,
        Status::Missing,
    ] {
        let _ = write!(
            legend,
            r#"<li><span class="swatch" style="background:{}"></span>{} <b>{}</b></li>"#,
            s.colour(),
            s.label(),
            count(s)
        );
    }
    let mut lists = String::new();
    for (title, status) in [
        ("Differs from the reference GPU", Status::Fails),
        (
            "Runs differently on the reference GPU's ISA",
            Status::Untestable,
        ),
        ("Not translated", Status::Missing),
    ] {
        let _ = write!(lists, "<h2>{title}</h2><dl>");
        for (name, cells) in &order {
            let names: Vec<String> = cells
                .iter()
                .filter(|f| f.status == status)
                .map(|f| {
                    format!(
                        r#"<code title="{}">{}</code>"#,
                        escape(&f.note),
                        escape(&f.name)
                    )
                })
                .collect();
            if !names.is_empty() {
                let _ = write!(lists, "<dt>{name}</dt><dd>{}</dd>", names.join(" "));
            }
        }
        lists.push_str("</dl>");
    }
    let translated = total - count(Status::Missing);
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU Instruction Status</title>
<style>
:root {{ --bg:#f7f7f5; --fg:#1d1d1b; --muted:#6b6b66; --line:#d8d8d2; --agrees:#2f9e44; --fails:#e03131; --untested:#f2c230; --untestable:#4c6ef5; --missing:#a3a39c; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141413; --fg:#ececea; --muted:#9a9a93; --line:#2c2c29; --agrees:#37b24d; --fails:#f03e3e; --untested:#f5c84c; --untestable:#5c7cfa; --missing:#55554f; }} }}
:root[data-theme="dark"] {{ --bg:#141413; --fg:#ececea; --muted:#9a9a93; --line:#2c2c29; --agrees:#37b24d; --fails:#f03e3e; --untested:#f5c84c; --untestable:#5c7cfa; --missing:#55554f; }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }} h2 {{ font-size:1.1rem; margin:28px 0 8px; }}
p {{ color:var(--muted); margin:4px 0 16px; }}
svg {{ width:100%; height:auto; display:block; }}
.group {{ fill:none; stroke:var(--line); stroke-width:1; }}
.label {{ font-size:11px; font-weight:600; fill:var(--fg); paint-order:stroke; stroke:var(--bg); stroke-width:3px; }}
ul.legend {{ list-style:none; padding:0; display:flex; flex-wrap:wrap; gap:6px 18px; }}
.swatch {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
dl {{ display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; }} dt {{ font-weight:600; }} dd {{ margin:0; }}
code {{ font-size:12px; margin-right:6px; white-space:nowrap; }}
</style></head><body><main>
<h1>GPU instruction status</h1>
<p>{translated} of {total} gfx1013 instruction forms translated. Hardware results: {}. The reference GPU is a Steam Deck's (Van Gogh, gfx1033, GFX10.3). The PS5's GPU is gfx1013 (GFX10.1): both are RDNA2 parts, but they differ in ISA generation and shader ABI, so green means the result matches gfx1033 and PS5 behaviour stays unverified. Blue marks forms GFX10.3 runs differently from GFX10.1, such as the multiply-adds it runs fused.</p>
<ul class="legend">{legend}</ul>
{svg}
{lists}
</main></body></html>
"#,
        escape(source)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_treemap_fills_its_area() {
        let r = squarify(
            &[6.0, 6.0, 4.0, 3.0, 2.0, 2.0, 1.0],
            Rect {
                x: 0.0,
                y: 0.0,
                w: 6.0,
                h: 4.0,
            },
        );
        let area: f64 = r.iter().map(|r| r.w * r.h).sum();
        assert!((area - 24.0).abs() < 1e-9);
        assert!(
            r.iter()
                .all(|r| r.x + r.w <= 6.0 + 1e-9 && r.y + r.h <= 4.0 + 1e-9)
        );
    }

    #[test]
    fn plain_value_differences_count_against_a_form() {
        let res = "v_add_f32\te32\tdiffer\t3 of 2048 lanes: 3 value\nv_add_f32\te64\tagree\nv_add_f32\te64 neg abs\tdiffer\t9 of 2048 lanes: 9 NaN bits\n";
        let forms = forms(res, "gfx1033");
        let get = |n: &str| forms.iter().find(|f| f.name == n).unwrap().status;
        assert_eq!(get("v_add_f32"), Status::Fails);
        assert_eq!(get("v_add_f32_e64"), Status::Fails);
        assert_eq!(get("v_sub_f32"), Status::Untested);
        assert_eq!(get("v_mad_f32"), Status::Untestable);
    }
}
