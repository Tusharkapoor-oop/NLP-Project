# Vendored browser libraries

Everything under this directory is checked in on purpose. The demo UI must
render on a machine with **no internet access**, and a CDN outage must never
turn the page into unstyled text. Each library is the official minified UMD
build, unmodified.

| File | Project | Version | License | Source |
| --- | --- | --- | --- | --- |
| `gsap.min.js` | GSAP core | 3.13.0 | GSAP standard "no charge" license | https://gsap.com |
| `ScrollTrigger.min.js` | GSAP ScrollTrigger | 3.13.0 | GSAP standard "no charge" license | https://gsap.com/docs/v3/Plugins/ScrollTrigger/ |
| `ScrollToPlugin.min.js` | GSAP ScrollToPlugin | 3.13.0 | GSAP standard "no charge" license | https://gsap.com/docs/v3/Plugins/ScrollToPlugin/ |
| `Flip.min.js` | GSAP Flip | 3.13.0 | GSAP standard "no charge" license | https://gsap.com/docs/v3/Plugins/Flip/ |
| `Observer.min.js` | GSAP Observer | 3.13.0 | GSAP standard "no charge" license | https://gsap.com/docs/v3/Plugins/Observer/ |
| `three.min.js` | Three.js | r160 (0.160.0) | MIT | https://threejs.org |
| `motion.min.js` | Motion One | 10.x | MIT | https://motion.dev |

## Notes

* **GSAP** is free for this use under its standard license (no charge for
  standard plugins). Review https://gsap.com/licensing/ before redistributing
  this demo commercially.
* `three.min.js` is the legacy UMD build, which prints a deprecation notice to
  the console from r150 onward. It is pinned deliberately: the page is a single
  classic-script setup with no bundler, and r160 is the last release that ships
  this build. A future migration to ES modules can drop the notice.
* **Motion One** exposes the `Motion` global; the page uses `Motion.animate`
  only, and degrades to CSS/GSAP when it is absent.
* No font files are vendored: the page uses system serif/sans/mono stacks so it
  stays fully offline.
