<script setup lang="ts">
import { useSlideContext } from '@slidev/client'
import gsap from 'gsap'
import { ref, watch } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// Slide 2's model and pulls, brought back. The pulls start at slide 2's angle and swing farther
// apart when the tasks send conflicting signals
const props = defineProps<{ first: number }>()

const angle = 60
const conflictAngle = 78
const reach = 200

function tip(length: number, degrees: number) {
  const radians = (degrees * Math.PI) / 180
  return { x: length * Math.sin(radians), y: length * Math.cos(radians) }
}

const start = tip(reach, angle)
const apart = tip(reach, conflictAngle)

// It shares a slide with slide 4's diagram, which leaves this study's encoder 256 wide, centred at
// x 180.33 and y 144, with an elbow arrow from its bottom to each output's swatch. The model starts
// there, drawn the same way, measured from the bottom of the box
const encoder = { x: -78 + (1040 - 256 - 393) / 3 + 256 / 2 - 442, y: 144 - 112 }
const elbow = { drop: 22, side: 34, tip: 98 }

// The model then slides left and its pulls shorten, as in slide 2's benchmark. The pulls are a
// little shorter than slide 2's, so the classification pull stays on the slide at full tug
const shift = 170 - 442
const shortReach = 130

// Back in the centre, the NER pull grows and tips the model, as on slide 2. When the model moves
// aside again for the post, its pulls shorten in the same proportion
const strongerReach = 300
const stronger = tip(strongerReach, conflictAngle)
const shortStrongerReach = (shortReach * strongerReach) / reach

const tilt = 5
const lean = tip(36, conflictAngle)
const pivot = { x: -32 * Math.sin((tilt * Math.PI) / 180), y: 32 * Math.cos((tilt * Math.PI) / 180) - 32 }

// Then the model splits in two, one trained with equal weighting and one with an advanced optimizer,
// each centred in its half of the slide. Gama and Grassi (2026, Appendix A.2.1) measured how far each
// task's gradient points from the two tasks' average direction, the dashed line: a cosine of about 0.7
// on the easier problems, where equal weighting kept up, Kurin et al.'s among them, and about 0.45 on
// the harder ones. The pulls take the angles those cosines stand for
// Their training cost is named at the foot of each line. The advanced optimizer needs about twice the
// time and memory: 735 against 323 seconds per epoch for IMTL on Kurin et al.'s (2022) two-task
// Cityscapes benchmark, and up to 11,980 against about 6,060 MiB in this study's pilot run
const sides = [
  { kind: 'equal', name: 'Equal weighting', x: 221 - 442, cost: ['Less time', 'and memory'] },
  { kind: 'advanced', name: 'Advanced optimizer', x: 663 - 442, cost: ['About twice the', 'time and memory'] },
]
const easierAngle = (Math.acos(0.7) * 180) / Math.PI
const harderAngle = (Math.acos(0.45) * 180) / Math.PI

// Zhu et al. (2025), Table 2, MiniLM: each multi-task score as a share of the single-task score.
// The axis starts at 95% of the single-task score, marked with a break, so the small drops show.
// The chart ends where slide 2's does but starts farther right, clear of the NER pull at full tug
const chart = { left: 410, width: 450, top: 104, base: 368 }
const full = chart.base - chart.top - 10
const bars = [
  { kind: 'classification', lines: ['Humanitarian', 'classification'], share: 63.6 / 63.9 },
  { kind: 'ner', lines: ['Named entity', 'recognition'], share: 76.4 / 77.1 },
].map((bar, index) => ({
  ...bar,
  x: chart.left + (chart.width * (index + 0.5)) / 2,
  height: (full * (bar.share - 0.95)) / 0.05,
}))

// The axis breaks between two parallel slashes near its foot
const gap = { top: chart.base - 34, bottom: chart.base - 24 }
const axis = `M ${chart.left} ${chart.top} V ${gap.top} M ${chart.left} ${gap.bottom} V ${chart.base} H ${chart.left + chart.width}`
const slashes = [gap.top, gap.bottom].map(y => `M ${chart.left - 8} ${y + 4} L ${chart.left + 8} ${y - 4}`).join(' ')

// The Taglish post from slide 4, ending where the chart ended. NER decides on every word, so each
// word gets its own bracket, and classification decides once, so one bracket runs under the post.
// The words' spans are measured in Inter at --text-base
const post = 'Baha na sa Marikina, need ng tubig!'
const words = [[0, 55], [61, 87], [94, 119], [125, 217], [229, 283], [290, 317], [324, 378]]
const postEnd = 384
const postX = chart.left + chart.width - postEnd
// The post, its brackets and the sentences span 23 above the baseline to 111 below it,
// so this centres them in the diagram's 456 height
const postY = 456 / 2 - (111 - 23) / 2

function bracket(start: number, end: number, top: number) {
  return `M ${start} ${top} V ${top + 6} H ${end} V ${top}`
}

// The legend's hatched swatch, spaced like the fitted dashes on slide 4: three stripes split the
// swatch's diagonal into four equal parts, so the middle stripe runs corner to corner
const swatch = 14
const stripes = [0.5, 1, 1.5].map((at) => {
  const c = swatch * at
  return c <= swatch ? `M 0 ${c} L ${c} 0` : `M ${c - swatch} ${swatch} L ${swatch} ${c - swatch}`
}).join(' ')

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const draw = { duration: 0.5, ease: 'power2.out' }
const moveShort = { duration: 0.5, ease: 'power2.inOut' }
const moveLong = { duration: 0.7, ease: 'power2.inOut' }

const root = ref<HTMLElement>()
const { $clicks } = useSlideContext()

let tugs: gsap.core.Timeline | undefined

// While the tasks conflict, they take turns pulling harder, and the model jerks toward whichever
// pulls, against the still line. This goes on beside the chart, until NER overpowers
function loop(clicks: number) {
  const step = clicks - props.first
  if (step === 2 || step === 3)
    tugs?.play()
  else
    tugs?.pause(0)
}

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.rig, .stand, .model-name, .pull .label', { opacity: 0 })
  gsap.set('.advanced, .top, .easier-problem, .harder-problem, .cost, .kurin, .gama, .pilot, .conflict, .overpower, .optimizer, .chart-text, .post-group, .counts, .zhu, .ali', { opacity: 0 })
  gsap.set('.axis', { strokeDasharray: 1, strokeDashoffset: 1 })
  gsap.set('.bar', { scaleY: 0, transformOrigin: '50% 100%' })
  // A tug stretches the arrow from the model outward, so its head moves and its root stays put
  gsap.set('.tug', { svgOrigin: '0 0' })
  tugs = gsap.timeline({ repeat: -1, repeatDelay: 0.6, paused: true })
    .to('.pull-ner .tug', { scaleY: 1.15, duration: 0.25, ease: 'power2.out' })
    .to('.sway', { x: 10, duration: 0.25, ease: 'power2.out' }, '<')
    .to('.pull-ner .tug', { scaleY: 1, duration: 0.3, ease: 'power2.inOut' })
    .to('.sway', { x: 0, duration: 0.3, ease: 'power2.inOut' }, '<')
    .to('.pull-classification .tug', { scaleY: 1.15, duration: 0.25, ease: 'power2.out' })
    .to('.sway', { x: -10, duration: 0.25, ease: 'power2.out' }, '<')
    .to('.pull-classification .tug', { scaleY: 1, duration: 0.3, ease: 'power2.inOut' })
    .to('.sway', { x: 0, duration: 0.3, ease: 'power2.inOut' }, '<')
  // Slide 4's look: the box where slide 4 left it, and each arm a thin ink arrow from the elbow
  // straight down to its output
  gsap.set('.rig', { x: encoder.x, y: encoder.y })
  gsap.set('.pull-classification .arm', { x: -elbow.side, y: elbow.drop, color: colour('ink'), svgOrigin: '0 0' })
  gsap.set('.pull-ner .arm', { x: elbow.side, y: elbow.drop, color: colour('ink'), svgOrigin: '0 0' })
  gsap.set('.shaft', { attr: { y2: elbow.tip - elbow.drop - 12 }, strokeWidth: 2 })
  gsap.set('.pull .head', { y: elbow.tip - elbow.drop, scaleX: 14 / 12, scaleY: 12 / 16, transformOrigin: '50% 100%' })
  gsap.set('.pull-classification .label', { x: 442 - start.x, y: 176 + start.y })
  gsap.set('.pull-ner .label', { x: 442 + start.x, y: 176 + start.y })

  timeline
    .addLabel('step0')
    // Slide 4's other elements leave, and the encoder becomes slide 2's multi-task model: the box
    // moves to the centre and is renamed, each elbow straightens and swings out into a pull, and
    // the pull takes its output's colour
    // It appears just after the label, at the same moment slide 4's copy is hidden
    .set('.rig', { opacity: 1 }, '+=0.01')
    .to('.rig', { x: 0, y: 0, ...moveLong }, '+=0.2')
    .to('.encoder-name', { opacity: 0, ...exit }, '<')
    .to('.model-name', { opacity: 1, ...enter }, '>')
    .to('.elbow', { scale: 0, svgOrigin: '0 0', ...moveLong }, '<-0.2')
    .to('.output', { opacity: 0, ...exit }, '<')
    .to('.pull-classification .arm', { x: 0, y: 0, rotation: angle, color: colour('classification'), ...moveLong }, '<')
    .to('.pull-ner .arm', { x: 0, y: 0, rotation: -angle, color: colour('ner'), ...moveLong }, '<')
    .to('.shaft', { attr: { y2: reach - 14 }, strokeWidth: 3, ...moveLong }, '<')
    .to('.pull .head', { y: reach, scaleX: 1, scaleY: 1, ...moveLong }, '<')
    .to('.stand, .pull .label', { opacity: 1, ...enter }, '>-0.2')
    .addLabel('step1')
    .to('.pull-classification .arm', { rotation: conflictAngle, svgOrigin: '0 0', ...moveShort })
    .to('.pull-ner .arm', { rotation: -conflictAngle, svgOrigin: '0 0', ...moveShort }, '<')
    .to('.pull-classification .label', { x: 442 - apart.x, y: 176 + apart.y, ...moveShort }, '<')
    .to('.pull-ner .label', { x: 442 + apart.x, y: 176 + apart.y, ...moveShort }, '<')
    .to('.conflict, .zhu', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step2')
    // The model moves aside as in slide 2's benchmark, and the chart compares it with the single-task models
    .to('.pull .label', { opacity: 0, ...exit })
    .to('.stand, .rig', { x: shift, ...moveLong }, '<')
    .to('.shaft', { attr: { y2: shortReach - 14 }, ...moveLong }, '<')
    .to('.pull .head', { y: shortReach, ...moveLong }, '<')
    .to('.axis', { strokeDashoffset: 0, ...draw }, '>-0.2')
    .to('.chart-text', { opacity: 1, ...enter }, '<0.2')
    .to('.bar', { scaleY: 1, ...draw, stagger: 0.08 }, '<')
    .addLabel('step3')
    // The chart leaves and the model returns to the centre, where NER overpowers classification
    // and tips the model, as on slide 2
    .to('.chart-text, .bar, .axis, .conflict, .zhu', { opacity: 0, ...exit })
    .to('.ali', { opacity: 1, ...enter }, '<0.2')
    .to('.stand, .rig', { x: 0, ...moveLong }, '<')
    .to('.shaft', { attr: { y2: reach - 14 }, ...moveLong }, '<')
    .to('.pull .head', { y: reach, ...moveLong }, '<')
    .to('.pull .label', { opacity: 1, ...enter }, '>-0.2')
    .to('.pull-ner .shaft', { attr: { y2: strongerReach - 14 }, ...moveShort }, '>')
    .to('.pull-ner .head', { y: strongerReach, ...moveShort }, '<')
    .to('.pull-ner .label', { x: 442 + stronger.x, y: 176 + stronger.y, ...moveShort }, '<')
    .to('.rig', { x: lean.x, y: lean.y, ...moveShort }, '<')
    .to('.model-box', { rotation: tilt, svgOrigin: '442 144', ...moveShort }, '<')
    .to('.pulls', { x: pivot.x, y: pivot.y, ...moveShort }, '<')
    .to('.overpower', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step4')
    // Why: the model moves aside again for a post. NER decides on every word, classification once
    .to('.pull .label', { opacity: 0, ...exit })
    .to('.stand', { x: shift, ...moveLong }, '<')
    .to('.rig', { x: shift + lean.x, ...moveLong }, '<')
    .to('.pull-classification .shaft', { attr: { y2: shortReach - 14 }, ...moveLong }, '<')
    .to('.pull-classification .head', { y: shortReach, ...moveLong }, '<')
    .to('.pull-ner .shaft', { attr: { y2: shortStrongerReach - 14 }, ...moveLong }, '<')
    .to('.pull-ner .head', { y: shortStrongerReach, ...moveLong }, '<')
    .to('.post-group', { opacity: 1, ...enter }, '>-0.2')
    .fromTo('.post-group .bracket.ner', { clipPath: 'inset(0% 50% 0% 50%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw, stagger: 0.06 }, '<0.2')
    .fromTo('.post-group .bracket.classification', { clipPath: 'inset(0% 50% 0% 50%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '>-0.2')
    .to('.counts', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step5')
    // The optimizer takes the place of NER overpowering. As on slide 2's last click, the post leaves,
    // the model returns to the centre and rights itself, and the pulls even out at slide 2's angle
    .to('.post-group, .overpower, .ali', { opacity: 0, ...exit })
    .to('.stand', { x: 0, ...moveLong }, '<')
    .to('.rig', { x: 0, y: 0, ...moveLong }, '<')
    .to('.model-box', { rotation: 0, svgOrigin: '442 144', ...moveLong }, '<')
    .to('.pulls', { x: 0, y: 0, ...moveLong }, '<')
    .to('.pull-classification .arm', { rotation: angle, svgOrigin: '0 0', ...moveLong }, '<')
    .to('.pull-ner .arm', { rotation: -angle, svgOrigin: '0 0', ...moveLong }, '<')
    .to('.shaft', { attr: { y2: reach - 14 }, ...moveLong }, '<')
    .to('.pull .head', { y: reach, ...moveLong }, '<')
    // The hidden labels wait at the pulls' new tips
    .set('.pull-classification .label', { x: 442 - start.x, y: 176 + start.y }, '<')
    .set('.pull-ner .label', { x: 442 + start.x, y: 176 + start.y }, '<')
    .to('.pull .label, .optimizer', { opacity: 1, ...enter }, `<${moveLong.duration - 0.2}`)
    .addLabel('step6')
    // Studies disagree. The model splits in two, one per optimizer, and on an easier problem the tasks
    // pull in nearly the same direction, so both models hang level
    .to('.pull .label, .optimizer', { opacity: 0, ...exit })
    .to('.advanced', { opacity: 1, ...enter }, '<')
    .to('.equal', { x: sides[0].x, ...moveLong }, '<')
    .to('.advanced', { x: sides[1].x, ...moveLong }, '<')
    .to('.pull-classification .arm', { rotation: easierAngle, svgOrigin: '0 0', ...moveLong }, '<')
    .to('.pull-ner .arm', { rotation: -easierAngle, svgOrigin: '0 0', ...moveLong }, '<')
    .to('.shaft', { attr: { y2: shortReach - 14 }, ...moveLong }, '<')
    .to('.pull .head', { y: shortReach, ...moveLong }, '<')
    .to('.top, .easier-problem, .kurin', { opacity: 1, ...enter }, '>-0.2')
    .addLabel('step7')
    // On a harder problem the pulls swing apart. With equal weighting NER overpowers and tips the
    // model, as before, while the advanced optimizer keeps its model level
    .to('.easier-problem, .kurin', { opacity: 0, ...exit })
    .to('.pull-classification .arm', { rotation: harderAngle, svgOrigin: '0 0', ...moveShort }, '<')
    .to('.pull-ner .arm', { rotation: -harderAngle, svgOrigin: '0 0', ...moveShort }, '<')
    .to('.equal .pull-ner .shaft', { attr: { y2: shortStrongerReach - 14 }, ...moveShort }, '>')
    .to('.equal .pull-ner .head', { y: shortStrongerReach, ...moveShort }, '<')
    .to('.equal .rig', { x: lean.x, y: lean.y, ...moveShort }, '<')
    .to('.equal .model-box', { rotation: tilt, svgOrigin: '442 144', ...moveShort }, '<')
    .to('.equal .pulls', { x: pivot.x, y: pivot.y, ...moveShort }, '<')
    .to('.harder-problem, .gama', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step8')
    // The two optimizers also differ in training cost, named at the foot of each line. The lines
    // shorten to give the two-line names room
    .to('.harder-problem, .gama', { opacity: 0, ...exit })
    .to('.midline', { attr: { d: 'M 442 56 V 362' }, ...moveShort }, '<')
    .to('.cost, .pilot', { opacity: 1, ...enter }, '>-0.2')
    .addLabel('step9')

  loop($clicks.value)
}, props.first)

watch($clicks, loop)
</script>

<template>
  <div ref="root" class="negative-transfer">
    <svg viewBox="0 0 884 456" width="884" height="456">
      <defs>
        <pattern v-for="bar in bars" :id="`hatch-${bar.kind}`" :key="bar.kind" :class="bar.kind" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="8" />
        </pattern>
        <clipPath id="legend-swatch">
          <rect :width="swatch" :height="swatch" rx="3" />
        </clipPath>
      </defs>

      <!-- One copy of the line and model per optimizer. They move as one until the model splits in two,
           so the advanced optimizer's copy stays hidden until then -->
      <g v-for="side in sides" :key="side.kind" class="side" :class="side.kind">
        <!-- The line and the label at its foot stay upright while the model tips. The label names the
             force on the model, and stays as long as that force is shown -->
        <g class="stand">
          <path class="midline" d="M 442 56 V 390" />
          <text class="tag top" x="442" y="36" text-anchor="middle">{{ side.name }}</text>
          <text class="tag conflict" x="442" y="424" text-anchor="middle">Conflicting signals</text>
          <text class="tag overpower" x="442" y="424" text-anchor="middle">NER overpowers</text>
          <text class="tag optimizer" x="442" y="424" text-anchor="middle">Multi-task optimizer</text>
          <text class="tag cost" text-anchor="middle">
            <tspan x="442" y="414">{{ side.cost[0] }}</tspan>
            <tspan x="442" y="436">{{ side.cost[1] }}</tspan>
          </text>
        </g>

        <g class="rig">
          <g class="sway">
            <g class="model-box">
              <rect class="model" x="314" y="112" width="256" height="64" rx="12" />
              <text class="name encoder-name" x="442" y="152" text-anchor="middle">Pretrained encoder</text>
              <text class="name model-name" x="442" y="152" text-anchor="middle">Multi-task model</text>
            </g>

            <g class="pulls">
              <g class="pull pull-classification">
                <g transform="translate(442 176)">
                  <path class="elbow" :d="`M 0 4 V ${elbow.drop} H ${-elbow.side}`" />
                  <rect class="output" :x="-elbow.side - 14" :y="elbow.tip + 6" width="28" height="28" rx="4" />
                  <g class="arm">
                    <g class="tug">
                      <line class="shaft" x1="0" y1="0" x2="0" :y2="reach - 14" />
                      <path class="head" :transform="`translate(0 ${reach})`" d="M 0 0 l -6 -16 h 12 z" />
                    </g>
                  </g>
                </g>
                <g class="label">
                  <text class="tag" text-anchor="middle">
                    <tspan x="0" y="34">Humanitarian</tspan>
                    <tspan x="0" y="56">classification</tspan>
                  </text>
                </g>
              </g>

              <g class="pull pull-ner">
                <g transform="translate(442 176)">
                  <path class="elbow" :d="`M 0 4 V ${elbow.drop} H ${elbow.side}`" />
                  <rect class="output" :x="elbow.side - 14" :y="elbow.tip + 6" width="28" height="28" rx="4" />
                  <g class="arm">
                    <g class="tug">
                      <line class="shaft" x1="0" y1="0" x2="0" :y2="reach - 14" />
                      <path class="head" :transform="`translate(0 ${reach})`" d="M 0 0 l -6 -16 h 12 z" />
                    </g>
                  </g>
                </g>
                <g class="label">
                  <text class="tag" text-anchor="middle">
                    <tspan x="0" y="34">Named entity</tspan>
                    <tspan x="0" y="56">recognition</tspan>
                  </text>
                </g>
              </g>
            </g>
          </g>
        </g>
      </g>

      <text class="tag easier-problem" x="442" y="424" text-anchor="middle">Easier problem</text>
      <text class="tag harder-problem" x="442" y="424" text-anchor="middle">Harder problem</text>

      <g class="chart">
        <g class="chart-text">
          <g class="hatched" :transform="`translate(${chart.left} ${chart.top - 50})`">
            <path :d="stripes" clip-path="url(#legend-swatch)" />
            <rect :width="swatch" :height="swatch" rx="3" />
          </g>
          <text class="tag" :x="chart.left + 22" :y="chart.top - 37">Single-task models</text>
          <rect class="swatch" :x="chart.left" :y="chart.top - 26" width="14" height="14" rx="3" />
          <text class="tag" :x="chart.left + 22" :y="chart.top - 13">Multi-task model</text>
          <text class="tag" text-anchor="middle" :transform="`translate(${chart.left - 16} ${(chart.top + chart.base) / 2}) rotate(-90)`">Higher is better</text>
          <text v-for="bar in bars" :key="bar.kind" class="tag" :class="bar.kind" text-anchor="middle">
            <tspan :x="bar.x" :y="chart.base + 30">{{ bar.lines[0] }}</tspan>
            <tspan :x="bar.x" :y="chart.base + 52">{{ bar.lines[1] }}</tspan>
          </text>
        </g>

        <g v-for="bar in bars" :key="bar.kind" :class="bar.kind">
          <rect class="bar single" :x="bar.x - 59" :y="chart.base - full" width="56" :height="full" :fill="`url(#hatch-${bar.kind})`" />
          <rect class="bar" :x="bar.x + 3" :y="chart.base - bar.height" width="56" :height="bar.height" />
        </g>

        <path class="axis" :d="axis" pathLength="1" />
        <!-- The axis starts near the single-task scores, not at zero -->
        <path class="chart-text axis-break" :d="slashes" />
      </g>


      <g class="post-group" :transform="`translate(${postX} ${postY})`">
        <text class="post">{{ post }}</text>
        <path v-for="[start, end] in words" :key="start" class="bracket ner" :d="bracket(start, end, 9)" />
        <path class="bracket classification" :d="bracket(0, postEnd, 27)" />
        <!-- What the brackets show, in Yngwie's words -->
        <g class="counts">
          <text class="sentence" :x="postEnd / 2" y="76" text-anchor="middle"><tspan class="ner">NER</tspan> decides once per word.</text>
          <text class="sentence" :x="postEnd / 2" y="106" text-anchor="middle"><tspan class="classification">Classification</tspan> decides once per post.</text>
        </g>
      </g>
    </svg>

    <footer class="zhu">Source: Zhu et al. (2025)</footer>
    <footer class="ali">Source: Ali et al. (2026)</footer>
    <footer class="kurin">Source: Kurin et al. (2022)</footer>
    <footer class="gama">Sources: Kurin et al. (2022), Gama and Grassi (2026)</footer>
    <footer class="pilot">Sources: Kurin et al. (2022), this study's pilot run</footer>
  </div>
</template>

<style scoped>
.negative-transfer {
  position: relative;
}

.post {
  font-size: var(--text-base);
}

.sentence {
  font-size: var(--text-sm);
  fill: var(--color-ink);
}

.sentence tspan {
  font-weight: 600;
  fill: currentColor;
}

.name {
  font-size: var(--text-base);
  font-weight: 600;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: currentColor;
}

path {
  fill: none;
  stroke: currentColor;
  stroke-width: 2;
}

.head {
  fill: currentColor;
  stroke: none;
}

.model {
  fill: var(--color-surface);
  stroke: currentColor;
  stroke-width: 2;
}

.shaft {
  stroke: currentColor;
  stroke-width: 3;
}

/* Slide 4's elbow arrows, in ink like the rest of slide 4's lines, and its output swatches */
.elbow {
  color: var(--color-ink);
}

.output {
  fill: currentColor;
}

.pull-ner,
.ner {
  color: var(--color-ner);
}

.pull-classification,
.classification {
  color: var(--color-classification);
}

.midline {
  color: var(--color-muted);
  stroke-dasharray: 8 6;
}

.stand,
.easier-problem,
.harder-problem,
.chart-text {
  color: var(--color-muted);
}

/* Single-task bars are hatched and outlined in their task's colour, the multi-task bars solid */
pattern line {
  stroke: currentColor;
  stroke-width: 3;
}

.bar:not(.single),
.swatch {
  fill: currentColor;
}

.bar.single {
  stroke: currentColor;
  stroke-width: 2;
}

.hatched path {
  stroke-width: 1.5;
}

.hatched rect {
  fill: none;
  stroke: currentColor;
  stroke-width: 1.5;
}

.axis,
.axis-break {
  color: var(--color-ink);
}

.bracket {
  stroke-width: 2.5;
}

/* The sources sit below the diagram, in the slide's bottom margin */
footer {
  position: absolute;
  left: 0;
  bottom: calc(-1 * var(--space-3));
  font-size: var(--text-2xs);
  color: var(--color-subtle);
}
</style>
