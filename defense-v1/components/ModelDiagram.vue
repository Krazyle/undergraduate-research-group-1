<script setup lang="ts">
import { useSlideContext } from '@slidev/client'
import gsap from 'gsap'
import { ref, watch } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The brackets and their labels are placed by this exact wording, set in Inter at --text-base
const post = 'Baha na dito sa Marikina, need ng tubig para sa mga bata. Pls help!'

const angle = 60
const reach = 200
const strongerReach = 320

function tip(length: number) {
  const radians = (angle * Math.PI) / 180
  return { x: length * Math.sin(radians), y: length * Math.cos(radians) }
}

const start = tip(reach)
const stronger = tip(strongerReach)

const tilt = 5
const lean = tip(36)
const pivot = { x: -32 * Math.sin((tilt * Math.PI) / 180), y: 32 * Math.cos((tilt * Math.PI) / 180) - 32 }

// The diagram slides left and its pulls shorten to make room for the benchmark chart
const shift = 150 - 442
const shortReach = 150

// Placeholder heights until the benchmark is run, as fractions of the chart's height
const negativeTransfer = [0.15, 0.7, 0.35, 0.55, 0.1, 0.8, 0.25, 0.6, 0.3, 0.9, 0.2, 0.65, 0.4]
const cost = [0.15, 0.35, 0.5, 0.3, 0.85, 0.4, 0.95, 0.25, 0.6, 0.45, 0.9, 0.55, 0.7]

const chart = { left: 350, width: 510, top: 104, base: 368 }
const columns = negativeTransfer.map((loss, index) => ({
  x: chart.left + (chart.width / negativeTransfer.length) * (index + 0.5),
  loss: loss * (chart.base - chart.top),
  cost: cost[index] * (chart.base - chart.top),
}))

// Motion tokens: things that enter settle in, things that leave speed away,
// and things that move on screen ease in and out, taking longer the farther they go
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const draw = { duration: 0.5, ease: 'power2.out' }
const moveShort = { duration: 0.5, ease: 'power2.inOut' }
const moveLong = { duration: 0.7, ease: 'power2.inOut' }

const svg = ref<SVGSVGElement>()
const { $clicks } = useSlideContext()

let decline: gsap.core.Tween | undefined

function loopDecline(step: number) {
  if (step >= 7)
    decline?.play()
  else
    decline?.pause(0)
}

useClickTimeline(svg, (timeline) => {
  gsap.set('text, .swatch, .marker, .midline, .tick', { opacity: 0 })
  decline = gsap.fromTo('.marker', { y: -3 }, { y: 3, duration: 0.6, ease: 'sine.inOut', repeat: -1, yoyo: true, paused: true })
  gsap.set('.model, .axis', { strokeDasharray: 1, strokeDashoffset: 1 })
  gsap.set('.bar', { scaleY: 0, transformOrigin: '50% 100%' })
  gsap.set('.result', { y: -110 })
  gsap.set('.result > .classification', { y: -54 })
  gsap.set('.key-classification', { y: -82 })

  timeline
    .addLabel('step0')
    .to('.example', { opacity: 1, ...enter })
    .addLabel('step1')
    .fromTo('.classification .bracket', { clipPath: 'inset(0% 50% 0% 50%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw })
    .to('.classification .tag, .key-classification > *', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step2')
    .to('.result > .classification, .key-classification', { y: 0, ...moveShort })
    .fromTo('.ner .bracket', { clipPath: 'inset(0% 50% 0% 50%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.3')
    .to('.ner .tag, .key-ner > *', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step3')
    .to('.result', { y: 0, ...moveLong })
    .to('.input', { opacity: 1, ...enter }, '<')
    .fromTo('.arrow-in', { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.3')
    .to('.model', { strokeDashoffset: 0, ...draw }, '>-0.2')
    .to('.name', { opacity: 1, ...enter }, '<0.2')
    .fromTo('.arrow-out', { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '>-0.1')
    .addLabel('step4')
    .to('.input, .arrow, .result', { opacity: 0, ...exit })
    .fromTo('.pull', { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.15')
    .to('.pull-classification .tag, .pull-ner .tag', { opacity: 1, ...enter }, '<0.3')
    .addLabel('step5')
    .to('.midline', { opacity: 1, ...enter })
    .to('.pull-ner .shaft', { attr: { y2: strongerReach - 14 }, ...moveShort }, '<')
    .to('.pull-ner .head', { y: strongerReach, ...moveShort }, '<')
    .to('.pull-ner .tag', { x: 442 + stronger.x, y: 176 + stronger.y, ...moveShort }, '<')
    .to('.rig', { x: lean.x, y: lean.y, ...moveShort }, '<')
    .to('.model-box', { rotation: tilt, svgOrigin: '442 144', ...moveShort }, '<')
    .to('.pulls', { x: pivot.x, y: pivot.y, ...moveShort }, '<')
    .addLabel('step6')
    .to('.decline > *', { opacity: 1, ...enter })
    .addLabel('step7')
    .to('.decline > *', { opacity: 0, ...exit })
    .to('.pull-ner .shaft', { attr: { y2: reach - 14 }, ...moveShort }, '<')
    .to('.pull-ner .head', { y: reach, ...moveShort }, '<')
    .to('.pull-ner .tag', { x: 442 + start.x, y: 176 + start.y, ...moveShort }, '<')
    .to('.rig', { x: 0, y: 0, ...moveShort }, '<')
    .to('.model-box', { rotation: 0, svgOrigin: '442 144', ...moveShort }, '<')
    .to('.pulls', { x: 0, y: 0, ...moveShort }, '<')
    .to('.optimizer', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step8')
    .to('.tick', { opacity: 1, ...enter })
    .addLabel('step9')
    .to('.pull-classification .tag, .pull-ner .tag', { opacity: 0, ...exit })
    .to('.rig, .midline', { x: shift, ...moveLong }, '<')
    .to('.shaft', { attr: { y2: shortReach - 14 }, ...moveLong }, '<')
    .to('.pull .head', { y: shortReach, ...moveLong }, '<')
    .to('.tick', { y: (shortReach - reach) / 2, ...moveLong }, '<')
    .to('.optimizer', { x: chart.left + chart.width / 2 - 442, ...moveLong }, '<')
    .to('.axis', { strokeDashoffset: 0, ...draw }, '<0.5')
    .to('.chart text, .chart .swatch', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step10')
    .to('.bar.loss', { scaleY: 1, ...draw, stagger: 0.03 })
    .addLabel('step11')
    .to('.bar.cost', { scaleY: 1, ...draw, stagger: 0.03 })
    .addLabel('step12')
    .to('.column:not(.kept)', { opacity: 0.2, ...exit })
    .to('.kept-name', { opacity: 1, ...enter }, '<')
    .addLabel('step13')

  loopDecline($clicks.value)
})

watch($clicks, loopDecline)
</script>

<template>
  <svg ref="svg" viewBox="0 0 884 456" width="884" height="456">
    <text class="post input" x="81" y="36">{{ post }}</text>

    <g class="arrow arrow-in">
      <path d="M 442 52 V 90" />
      <path class="head" d="M 442 102 l -7 -12 h 14 z" />
    </g>

    <path class="midline" d="M 442 56 V 390" />
    <!-- Negative transfer sits at the foot of the line, and the optimizer that reduces it takes its place -->
    <g class="decline">
      <path class="marker" d="M 311 410 h 18 l -9 16 z" />
      <text class="tag" x="442" y="424" text-anchor="middle">Negative transfer</text>
    </g>
    <text class="tag optimizer" x="442" y="424" text-anchor="middle">Multi-task optimizer</text>
    <text class="tag kept-name" x="150" y="424" text-anchor="middle">Optimizer 1</text>

    <g class="rig">
      <g class="model-box">
        <rect class="model" x="314" y="112" width="256" height="64" rx="12" pathLength="1" />
        <text class="name" x="442" y="152" text-anchor="middle">Multi-task model</text>
      </g>

      <g class="pulls">
        <g class="pull-classification">
          <g class="pull" :transform="`translate(442 176) rotate(${angle})`">
            <line class="shaft" x1="0" y1="0" x2="0" :y2="reach - 14" />
            <line class="tick" x1="-9" :y1="reach / 2 - 5" x2="9" :y2="reach / 2 - 5" />
            <line class="tick" x1="-9" :y1="reach / 2 + 5" x2="9" :y2="reach / 2 + 5" />
            <path class="head" :transform="`translate(0 ${reach})`" d="M 0 0 l -6 -16 h 12 z" />
          </g>
          <text class="tag" text-anchor="middle" :transform="`translate(${442 - start.x} ${176 + start.y})`">
            <tspan x="0" y="34">Humanitarian</tspan>
            <tspan x="0" y="56">classification</tspan>
          </text>
        </g>

        <g class="pull-ner">
          <g class="pull" :transform="`translate(442 176) rotate(${-angle})`">
            <line class="shaft" x1="0" y1="0" x2="0" :y2="reach - 14" />
            <line class="tick" x1="-9" :y1="reach / 2 - 5" x2="9" :y2="reach / 2 - 5" />
            <line class="tick" x1="-9" :y1="reach / 2 + 5" x2="9" :y2="reach / 2 + 5" />
            <path class="head" :transform="`translate(0 ${reach})`" d="M 0 0 l -6 -16 h 12 z" />
          </g>
          <text class="tag" text-anchor="middle" :transform="`translate(${442 + start.x} ${176 + start.y})`">
            <tspan x="0" y="34">Named entity</tspan>
            <tspan x="0" y="56">recognition</tspan>
          </text>
        </g>
      </g>
    </g>

    <g class="arrow arrow-out">
      <path d="M 442 186 V 224" />
      <path class="head" d="M 442 236 l -7 -12 h 14 z" />
    </g>

    <g class="result">
      <text class="post example" x="81" y="264">{{ post }}</text>

      <g class="ner">
        <path class="bracket" d="M 253 273 V 279 H 345 V 273" />
        <path class="bracket" d="M 357 273 V 279 H 506 V 273" />
        <text class="tag" x="299" y="305" text-anchor="middle">Location</text>
        <text class="tag" x="431.5" y="305" text-anchor="middle">Request</text>
      </g>

      <g class="classification">
        <path class="bracket" d="M 81 327 V 333 H 803 V 327" />
        <text class="tag" x="442" y="361" text-anchor="middle">Requests or urgent needs</text>
      </g>

      <g class="legend">
        <g class="key-ner">
          <rect class="swatch ner" x="286" y="389" width="14" height="14" rx="3" />
          <text class="tag" x="308" y="402">Named entity recognition</text>
        </g>
        <g class="key-classification">
          <rect class="swatch classification" x="271" y="417" width="14" height="14" rx="3" />
          <text class="tag" x="293" y="430">Humanitarian classification</text>
        </g>
      </g>
    </g>
    <g class="chart">
      <rect class="swatch loss" :x="chart.left" :y="chart.top - 50" width="14" height="14" rx="3" />
      <text class="tag" :x="chart.left + 22" :y="chart.top - 37">Negative transfer</text>
      <rect class="swatch cost" :x="chart.left" :y="chart.top - 26" width="14" height="14" rx="3" />
      <text class="tag" :x="chart.left + 22" :y="chart.top - 13">Training time and memory</text>

      <path class="axis" :d="`M ${chart.left} ${chart.top} V ${chart.base} H ${chart.left + chart.width}`" pathLength="1" />
      <text class="tag" text-anchor="middle" :transform="`translate(${chart.left - 16} ${(chart.top + chart.base) / 2}) rotate(-90)`">Lower is better</text>

      <g v-for="(column, index) in columns" :key="index" class="column" :class="{ kept: index === 0 }">
        <rect class="bar loss" :x="column.x - 15" :y="chart.base - column.loss" width="14" :height="column.loss" />
        <rect class="bar cost" :x="column.x + 1" :y="chart.base - column.cost" width="14" :height="column.cost" />
        <text class="number" :x="column.x" :y="chart.base + 28" text-anchor="middle">{{ index + 1 }}</text>
      </g>
    </g>
  </svg>
</template>

<style scoped>
.post {
  font-size: var(--text-base);
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

.model,
path {
  fill: none;
  stroke: currentColor;
  stroke-width: 2;
}

.head,
.marker {
  fill: currentColor;
  stroke: none;
}

.bracket {
  stroke-width: 2.5;
}

.shaft {
  stroke: currentColor;
  stroke-width: 3;
}

.tick {
  stroke: currentColor;
  stroke-width: 2.5;
}

.ner,
.pull-ner {
  color: var(--color-ner);
}

.classification,
.pull-classification {
  color: var(--color-classification);
}

.model {
  fill: var(--color-surface);
}

.midline {
  color: var(--color-muted);
  stroke-dasharray: 8 6;
}

.legend,
.decline,
.optimizer,
.kept-name,
.chart .tag {
  color: var(--color-muted);
}

.marker {
  color: var(--color-negative);
}

.swatch,
.bar {
  fill: currentColor;
}

.loss {
  color: var(--color-negative);
}

.cost {
  color: var(--color-ink);
}

.number {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: currentColor;
}
</style>
