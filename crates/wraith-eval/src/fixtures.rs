//! Ground-truth fixtures for the eval harness (roadmap M0).
//!
//! Each fixture is an SVG (the *ground truth*), rasterised with resvg at the
//! working size. Evaluating against the ground-truth **SVG rendered at 1× and
//! 4×** measures true vectorisation fidelity — sub-pixel boundary placement —
//! not just reproduction of the input pixels (prototype §5).

/// One named ground-truth fixture.
pub struct Fixture {
    /// Fixture name (file stem, report key).
    pub name: &'static str,
    /// Ground-truth SVG source.
    pub svg: String,
}

/// The four prototype fixtures plus an icon set for the M4 primitive gate.
#[must_use]
pub fn fixtures() -> Vec<Fixture> {
    let head = r#"<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">"#;
    let flat = format!(
        r##"{head}<rect width="512" height="512" fill="#fff"/>
<circle cx="256" cy="256" r="200" fill="#1e5ac8"/>
<rect x="110" y="110" width="190" height="190" rx="24" fill="#dc2828" transform="rotate(12 205 205)"/>
<polygon points="330,120 360,200 445,205 380,255 402,335 330,290 258,335 280,255 215,205 300,200" fill="#fad23c"/>
<path d="M120 380 C 200 300, 320 460, 400 380" stroke="#111" stroke-width="14" fill="none" stroke-linecap="round"/>
<path d="M60 60 H 140" stroke="#111" stroke-width="1.5"/>
<circle cx="256" cy="256" r="60" fill="#fff"/></svg>"##
    );
    let gradient = format!(
        r##"{head}<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ff5f6d"/><stop offset="1" stop-color="#ffc371"/></linearGradient>
<radialGradient id="r"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#2b6cb0"/></radialGradient></defs>
<rect width="512" height="512" fill="#fff"/><circle cx="256" cy="256" r="210" fill="url(#g)"/>
<rect x="150" y="150" width="212" height="212" rx="40" fill="url(#r)"/>
<rect x="40" y="440" width="432" height="30" fill="#222"/></svg>"##
    );
    let illustration = format!(
        r##"{head}<rect width="512" height="512" fill="#bfe3f5"/>
<circle cx="400" cy="90" r="48" fill="#ffd54a"/>
<ellipse cx="120" cy="90" rx="70" ry="26" fill="#f7fbff"/><ellipse cx="160" cy="75" rx="50" ry="24" fill="#f7fbff"/>
<path d="M0 330 Q 130 220 260 330 T 512 300 V512 H0Z" fill="#6bb36b"/>
<path d="M0 400 Q 180 310 340 400 T 512 380 V512 H0Z" fill="#3f8f4f"/>
<rect x="90" y="300" width="150" height="110" fill="#e8d3a7"/><polygon points="75,300 165,225 255,300" fill="#b5483a"/>
<rect x="150" y="350" width="34" height="60" fill="#7a4b2a"/><rect x="104" y="322" width="30" height="26" fill="#8fd0f0" stroke="#fff" stroke-width="3"/>
<rect x="378" y="330" width="22" height="80" fill="#6b4226"/><circle cx="389" cy="300" r="46" fill="#2f7d3b"/><circle cx="360" cy="325" r="30" fill="#378a44"/><circle cx="420" cy="325" r="30" fill="#378a44"/>
<circle cx="300" cy="360" r="12" fill="#e53935"/><circle cx="322" cy="372" r="9" fill="#e53935"/></svg>"##
    );
    let transparent = format!(
        r##"{head}<circle cx="256" cy="256" r="220" fill="#0b3d91"/><polygon points="256,90 300,210 430,215 330,295 365,420 256,350 147,420 182,295 82,215 212,210" fill="#fff"/><path d="M40 256 H 472" stroke="#fc3d21" stroke-width="3"/></svg>"##
    );
    // Icon set (M4 gate): rounded rects, circles, ellipses, straight edges.
    let icons = format!(
        r##"{head}<rect width="512" height="512" fill="#fff"/>
<rect x="40" y="40" width="200" height="200" rx="36" fill="#2f6f3f"/>
<circle cx="360" cy="140" r="100" fill="#2456a8"/>
<ellipse cx="140" cy="380" rx="110" ry="60" fill="#8a3fb0"/>
<rect x="260" y="300" width="212" height="160" rx="48" fill="#c2571f"/>
<rect x="248" y="120" width="80" height="80" fill="#1c1c1c"/></svg>"##
    );
    vec![
        Fixture {
            name: "flat_logo",
            svg: flat,
        },
        Fixture {
            name: "gradient_logo",
            svg: gradient,
        },
        Fixture {
            name: "illustration",
            svg: illustration,
        },
        Fixture {
            name: "transparent_logo",
            svg: transparent,
        },
        Fixture {
            name: "icons",
            svg: icons,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_are_well_formed_svgs() {
        for f in fixtures() {
            assert!(f.svg.starts_with("<svg"));
            assert!(f.svg.ends_with("</svg>"));
            assert!(f.svg.contains("http://www.w3.org/2000/svg"));
        }
        assert_eq!(fixtures().len(), 5);
    }
}
