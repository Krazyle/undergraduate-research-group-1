<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// Part 1: Figure 3.1 of Chapter 3, with the same steps, stores, arrows and phases, drawn the way the
// deck draws its diagrams rather than as on paper. Its three phases become three equal columns under
// captions, and every part, the posts going in and out included, is a box like the deck's model box. Each label is cut down to
// the name of its step, since Chapter 3 gives the details. Each part names the click it appears on,
// in the order Prinz describes them
type Kind = 'box'
interface Node { kind: Kind, step: number, x: number, y: number, width: number, height: number, label: string }

// The slide's content area is 884 wide: three 268-wide columns with two 40-wide gutters
const columns = [134, 442, 750]
const [left, middle, right] = columns
const width = 220
const height = 34
const box = (step: number, x: number, y: number, label: string, size = { width, height }) =>
  ({ kind: 'box' as const, step, x, y, label, ...size })
const fork = { width: 124, height: 50 }

// The boxes in each column are 36 apart, room for an arrow that reads as a line and not just a
// head. Each column starts under the captions, and the middle one, the longest, sets the height
const gap = 36
const top = 64
const row = (index: number) => top + index * (height + gap)
const saved = row(1) + fork.height + gap
const below = (y: number) => y + height + gap
const selection = below(saved)
const evaluation = below(selection)
const finalTop = below(evaluation)
const bottom = finalTop + height

const nodes: Node[] = [
  box(2, left, row(0), 'Posts'),
  box(2, left, row(1), 'Cleaning'),
  box(2, left, row(2), 'Labeling'),
  box(2, left, row(3), 'Splitting'),
  box(2, left, row(4), 'Labeled dataset'),

  box(3, middle, row(0), 'Fine-tuning'),
  box(3, middle - 68, row(1), 'Single-task\nmodels', fork),
  box(3, middle + 68, row(1), 'Candidate\nmodels', fork),
  box(3, middle, saved, 'Saved models'),
  box(4, middle, selection, 'Optimizer selection'),
  box(4, middle, evaluation, 'Final evaluation'),

  box(5, right, row(0), 'Database'),
  box(5, right, row(1), 'API server'),
  box(5, right, row(2), 'Web interface'),
  box(5, right, row(3), 'Sorted posts\nfor DRRMO staff', { width, height: 50 }),
]

const captions = [
  { step: 1, x: left, lines: ['Data', 'preparation'] },
  { step: 1, x: middle, lines: ['Model training', 'and selection'] },
  { step: 1, x: right, lines: ['Prototype web', 'application'] },
]

// Each arrow runs through its points and ends in a head, or a head at both ends. The arrows between
// columns run down the middle of the gutters, and fine-tuning splits in two and joins again with
// slide 4's elbows
const down = (x: number, from: number, to: number) => [[x, from], [x, to]]
const split = top + height + gap / 2
// The final model keeps slide 2's proportions at the height of the other boxes
const finalWidth = (256 * height) / 64
const arrows = [
  { step: 2, points: down(left, row(0) + height, row(1)) },
  { step: 2, points: down(left, row(1) + height, row(2)) },
  { step: 2, points: down(left, row(2) + height, row(3)) },
  { step: 2, points: down(left, row(3) + height, row(4)) },
  { step: 3, points: [[left + width / 2, row(4) + height / 2], [288, row(4) + height / 2], [288, top + height / 2], [middle - width / 2, top + height / 2]] },
  { step: 3, points: [[middle, top + height], [middle, split], [middle - 68, split], [middle - 68, row(1)]] },
  { step: 3, points: [[middle, top + height], [middle, split], [middle + 68, split], [middle + 68, row(1)]] },
  { step: 3, points: [[middle - 68, row(1) + fork.height], [middle - 68, saved - gap / 2], [middle, saved - gap / 2], [middle, saved]] },
  { step: 3, points: [[middle + 68, row(1) + fork.height], [middle + 68, saved - gap / 2], [middle, saved - gap / 2], [middle, saved]] },
  { step: 4, points: down(middle, saved + height, selection) },
  { step: 4, points: down(middle, selection + height, evaluation) },
  { step: 4, points: down(middle, evaluation + height, finalTop) },
  { step: 5, points: [[middle + finalWidth / 2, finalTop + height / 2], [596, finalTop + height / 2], [596, row(1) + height / 2], [right - width / 2, row(1) + height / 2]] },
  { step: 5, points: down(right, row(0) + height, row(1)), both: true },
  { step: 5, points: down(right, row(1) + height, row(2)), both: true },
  { step: 5, points: down(right, row(2) + height, row(3)) },
]

// The deck's arrowhead, 14 wide and 12 long, turned to the line's last direction
const headLength = 12
function head(from: number[], to: number[]) {
  const angle = (Math.atan2(to[1] - from[1], to[0] - from[0]) * 180) / Math.PI - 90
  return `translate(${to[0]} ${to[1]}) rotate(${angle})`
}
function shorten(from: number[], to: number[]) {
  const length = Math.hypot(to[0] - from[0], to[1] - from[1])
  const k = (length - headLength) / length
  return [from[0] + (to[0] - from[0]) * k, from[1] + (to[1] - from[1]) * k]
}
const drawn = arrows.map((arrow) => {
  const points = arrow.points.map(point => [...point])
  const ends = [{ from: points[points.length - 2], to: points[points.length - 1] }]
  points[points.length - 1] = shorten(points[points.length - 2], points[points.length - 1])
  if (arrow.both) {
    ends.push({ from: points[1], to: points[0] })
    points[0] = shorten(points[1], points[0])
  }
  return { ...arrow, d: `M ${points.map(point => point.join(' ')).join(' L ')}`, heads: ends.map(end => head(end.from, end.to)) }
})

// The final model is the multi-task model of slides 2 and 4, its two task heads pulling at slide 2's
// angle, drawn where slide 2 draws it. It shrinks into the foot of the training column, leaving its
// task heads behind until part 2, where it grows back and they draw in again
const model = { x: 442, y: 144 }
// The figure is drawn from y = 0 to its bottom box, then shrunk a little and centred on the slide's
// content area, so it has room above and below. The final model lands on its place in the shrunk
// figure, as tall as its boxes
const figureScale = 0.92
const drawnCentre = { x: 442, y: bottom / 2 }
const centre = { x: 442, y: 228 }
const figureTransform = `translate(${centre.x} ${centre.y}) scale(${figureScale}) translate(${-drawnCentre.x} ${-drawnCentre.y})`
const small = {
  x: centre.x + (middle - drawnCentre.x) * figureScale,
  y: centre.y + (finalTop + height / 2 - drawnCentre.y) * figureScale,
  scale: (height / 64) * figureScale,
}
const angle = 60
const reach = 200
const start = { x: reach * Math.sin((angle * Math.PI) / 180), y: reach * Math.cos((angle * Math.PI) / 180) }
const pulls = [
  { kind: 'classification', rotate: angle, x: model.x - start.x, lines: ['Humanitarian', 'classification'] },
  { kind: 'ner', rotate: -angle, x: model.x + start.x, lines: ['Named entity', 'recognition'] },
]

// Part 2: the model itself, reading slide 2's post once. The optimizer is not drawn, as Prinz says
const post = 'Baha na dito sa Marikina, need ng tubig para sa mga bata. Pls help!'

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const draw = { duration: 0.5, ease: 'power2.out' }
const moveLong = { duration: 0.7, ease: 'power2.inOut' }

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.part, .head, .short-name, .input', { opacity: 0 })
  gsap.set('.line', { strokeDasharray: 1, strokeDashoffset: 1 })
  gsap.set('.final', { svgOrigin: `${model.x} ${model.y}` })

  timeline
    .addLabel('step0')
    // The model moves to its place in the figure, and the figure's three phases are named
    .to('.final', { x: small.x - model.x, y: small.y - model.y, scale: small.scale, ...moveLong })
    .to('.long-name, .pull, .pull-name', { opacity: 0, ...exit }, '<')
    .to('.short-name', { opacity: 1, ...enter }, '<0.3')
    .to('.caption', { opacity: 1, stagger: 0.08, ...enter }, '<0.2')
    .addLabel('step1')

  // Each phase brings in its boxes in reading order, and draws the arrows between them
  for (let step = 2; step <= 5; step++) {
    timeline
      .to(`.node.step-${step}`, { opacity: 1, stagger: 0.08, ...enter })
      .to(`.arrow.step-${step}`, { opacity: 1, duration: 0 }, '<')
      .to(`.arrow.step-${step} .line`, { strokeDashoffset: 0, stagger: 0.08, ...draw }, '<0.1')
      .to(`.arrow.step-${step} .head`, { opacity: 1, stagger: 0.08, ...enter }, '<0.3')
      .addLabel(`step${step}`)
  }

  timeline
    // The figure leaves, and the final model grows back to the size and place it has on slide 2,
    // reading the post
    .to('.part', { opacity: 0, ...exit })
    .to('.final', { x: 0, y: 0, scale: 1, ...moveLong }, '<0.1')
    .to('.short-name', { opacity: 0, ...exit }, '<')
    .to('.long-name', { opacity: 1, ...enter }, '<0.3')
    .to('.input', { opacity: 1, ...enter }, '>-0.1')
    .fromTo('.arrow-in', { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.2')
    .addLabel('step6')
    // Its two task heads, one for each task, draw in as they first did on slide 2. They start just
    // after the label, so seeking to step6 leaves them hidden
    .fromTo('.pull', { opacity: 1, clipPath: 'inset(0% 0% 100% 0%)' }, { opacity: 1, clipPath: 'inset(0% 0% 0% 0%)', immediateRender: false, ...draw }, '+=0.01')
    .to('.pull-name', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step7')
})
</script>

<template>
  <div ref="root" class="system-diagram">
    <svg viewBox="0 0 884 456" width="884" height="456">
      <g :transform="figureTransform">
        <text v-for="caption in captions" :key="caption.x" class="part caption tag" :class="`step-${caption.step}`" text-anchor="middle">
          <tspan v-for="(line, row) in caption.lines" :key="row" :x="caption.x" :y="18 + row * 22">{{ line }}</tspan>
        </text>

        <g v-for="(arrow, index) in drawn" :key="index" class="part arrow" :class="`step-${arrow.step}`">
          <path class="line" :d="arrow.d" pathLength="1" />
          <path v-for="(transform, end) in arrow.heads" :key="end" class="head" d="M 0 0 l -7 -12 h 14 z" :transform="transform" />
        </g>

        <g v-for="node in nodes" :key="node.label" class="part node" :class="`step-${node.step}`">
          <rect v-if="node.kind === 'box'" class="model" :x="node.x - node.width / 2" :y="node.y" :width="node.width" :height="node.height" rx="12" />
          <foreignObject :x="node.x - node.width / 2" :y="node.y" :width="node.width" :height="node.height">
            <div class="label" :class="node.kind">{{ node.label }}</div>
          </foreignObject>
        </g>

      </g>

      <!-- Part 2, drawn with slide 2's measurements -->
      <text class="post input" x="81" y="36">{{ post }}</text>
      <g class="input arrow-in">
        <path class="plain" d="M 442 52 V 90" />
        <path class="tip" d="M 442 102 l -7 -12 h 14 z" />
      </g>
      <!-- The final model, the model of slides 2 and 4. Its short name is set so it reads at --text-xs,
           in the weight of the steps, once the model has shrunk into the figure -->
      <g class="final">
        <g v-for="pull in pulls" :key="pull.kind" :class="pull.kind">
          <g class="pull" :transform="`translate(${model.x} 176) rotate(${pull.rotate})`">
            <line class="shaft" x1="0" y1="0" x2="0" :y2="reach - 14" />
            <path class="pull-head" :transform="`translate(0 ${reach})`" d="M 0 0 l -6 -16 h 12 z" />
          </g>
          <text class="tag pull-name" text-anchor="middle" :transform="`translate(${pull.x} ${176 + start.y})`">
            <tspan v-for="(line, row) in pull.lines" :key="row" x="0" :y="34 + row * 22">{{ line }}</tspan>
          </text>
        </g>
        <rect class="model" x="314" y="112" width="256" height="64" rx="12" />
        <text class="name long-name" x="442" y="152" text-anchor="middle">Multi-task model</text>
        <text class="name short-name" x="442" y="153" text-anchor="middle">Final model</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: currentColor;
}

.caption {
  font-size: calc(var(--text-2xs) / 0.92);
  color: var(--color-muted);
}

.post {
  font-size: var(--text-base);
}

.name {
  font-size: var(--text-base);
  font-weight: 600;
}

.short-name {
  font-size: calc(var(--text-xs) * 64 / 34);
  font-weight: 400;
}

.line,
.plain {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}

.head,
.tip {
  fill: var(--color-ink);
}

.model {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}

.label {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: var(--text-xs);
  line-height: 1.2;
  text-align: center;
  white-space: pre-line;
  color: var(--color-ink);
}

.shaft {
  stroke: currentColor;
  stroke-width: 3;
}

.pull-head {
  fill: currentColor;
}

.ner {
  color: var(--color-ner);
}

.classification {
  color: var(--color-classification);
}
</style>
