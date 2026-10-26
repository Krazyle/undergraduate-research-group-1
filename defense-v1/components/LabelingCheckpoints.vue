<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit } from '../shared/motion'

const dim = 0.3

const lanes = [
  { key: 'data', tag: 'Data', start: 'Cleaned posts', question: ['Enough', 'posts?'], backup: 'Add Facebook posts', y: 66 },
  { key: 'labeling', tag: 'Labeling', start: 'Practice batch', question: ['Annotators', 'agree?'], backup: 'Revise the guidelines', y: 256 },
]

const start = { width: 220, height: 60 }
const diamond = { x: 392, halfWidth: 100, halfHeight: 50 }
const backup = { x: 277, width: 230, height: 46, gap: 80 }
const finish = { x: 692, y: 26, width: 192, height: 270 }

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.question, .branch', { opacity: 0 })

  timeline.addLabel('step0')

  timeline
    .to('.lane.labeling', { opacity: dim, ...exit })
    .to('.data .diamond', { stroke: colour('ink'), ...enter }, '<')
    .to('.data .question', { opacity: 1, ...enter }, '<0.1')
    .to('.data .branch', { opacity: 1, ...enter }, '>')
    .addLabel('step1')

  timeline
    .to('.lane.data', { opacity: dim, ...exit })
    .to('.lane.labeling', { opacity: 1, ...enter }, '<')
    .to('.labeling .diamond', { stroke: colour('ink'), ...enter }, '<')
    .to('.labeling .question', { opacity: 1, ...enter }, '<0.1')
    .to('.labeling .branch', { opacity: 1, ...enter }, '>')
    .addLabel('step2')

  timeline
    .to('.lane.data', { opacity: 1, ...enter })
    .to('.yes-line', { stroke: colour('positive'), ...enter }, '<')
    .to('.yes-head, .yes-text', { fill: colour('positive'), ...enter }, '<')
    .to('.finish-box', { fill: colour('positive'), stroke: colour('positive'), ...enter }, '>-0.1')
    .to('.finish-text', { fill: colour('surface'), ...enter }, '<')
    .addLabel('step3')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 384" width="884" height="384">
    <g v-for="lane in lanes" :key="lane.key" class="lane" :class="lane.key">
      <text class="tag" x="0" :y="lane.y - 44">{{ lane.tag }}</text>

      <rect class="box" x="0" :y="lane.y - start.height / 2" :width="start.width" :height="start.height" rx="12" />
      <text class="box-text" :x="start.width / 2" :y="lane.y + 7" text-anchor="middle">{{ lane.start }}</text>
      <path class="line" :d="`M ${start.width} ${lane.y} H ${diamond.x - diamond.halfWidth - 10}`" />
      <path class="head" :d="`M ${diamond.x - diamond.halfWidth} ${lane.y} l -11 -6 v 12 z`" />

      <path
        class="diamond"
        :d="`M ${diamond.x} ${lane.y - diamond.halfHeight} L ${diamond.x + diamond.halfWidth} ${lane.y} L ${diamond.x} ${lane.y + diamond.halfHeight} L ${diamond.x - diamond.halfWidth} ${lane.y} Z`"
      />
      <g class="question">
        <text class="box-text" :x="diamond.x" :y="lane.y - 4" text-anchor="middle">{{ lane.question[0] }}</text>
        <text class="box-text" :x="diamond.x" :y="lane.y + 20" text-anchor="middle">{{ lane.question[1] }}</text>
      </g>

      <g class="branch">
        <path class="line" :d="`M ${diamond.x} ${lane.y + diamond.halfHeight} V ${lane.y + backup.gap - 10}`" />
        <path class="head" :d="`M ${diamond.x} ${lane.y + backup.gap} l -6 -11 h 12 z`" />
        <text class="small" :x="diamond.x + 10" :y="lane.y + 70">No</text>
        <rect class="box" :x="backup.x" :y="lane.y + backup.gap" :width="backup.width" :height="backup.height" rx="12" />
        <text class="backup-text" :x="backup.x + backup.width / 2" :y="lane.y + backup.gap + 30" text-anchor="middle">{{ lane.backup }}</text>
        <path class="line" :d="`M ${backup.x} ${lane.y + backup.gap + backup.height / 2} H ${start.width / 2} V ${lane.y + start.height / 2 + 10}`" />
        <path class="head" :d="`M ${start.width / 2} ${lane.y + start.height / 2} l -6 11 h 12 z`" />
        <text class="small" :x="start.width / 2 + 10" :y="lane.y + 74">Check again</text>
      </g>

      <path class="yes-line" :d="`M ${diamond.x + diamond.halfWidth} ${lane.y} H ${finish.x - 10}`" />
      <path class="yes-head" :d="`M ${finish.x} ${lane.y} l -11 -6 v 12 z`" />
      <text class="small yes-text" :x="(diamond.x + diamond.halfWidth + finish.x) / 2" :y="lane.y - 12" text-anchor="middle">Yes</text>
    </g>

    <rect class="finish-box" :x="finish.x" :y="finish.y" :width="finish.width" :height="finish.height" rx="12" />
    <text class="finish-text title" :x="finish.x + finish.width / 2" :y="finish.y + finish.height / 2 - 4" text-anchor="middle">Full labeling</text>
    <text class="finish-text sub" :x="finish.x + finish.width / 2" :y="finish.y + finish.height / 2 + 26" text-anchor="middle">starts in December</text>
  </svg>
</template>

<style scoped>
svg {
  overflow: visible;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  fill: var(--color-muted);
}

.box,
.diamond {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.diamond {
  stroke: var(--color-faint);
}

.box-text {
  font-size: var(--text-xs);
  font-weight: 700;
  fill: var(--color-ink);
}

.backup-text {
  font-size: var(--text-2xs);
  font-weight: 700;
  fill: var(--color-ink);
}

.line {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.head {
  fill: var(--color-ink);
}

.small {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-muted);
}

.yes-line {
  fill: none;
  stroke: var(--color-faint);
  stroke-width: 2.5;
}

.yes-head,
.small.yes-text {
  fill: var(--color-faint);
}

.finish-box {
  fill: var(--color-surface);
  stroke: var(--color-faint);
  stroke-width: 2;
}

.finish-text {
  fill: var(--color-faint);
}

.finish-text.title {
  font-size: var(--text-md);
  font-weight: 700;
}

.finish-text.sub {
  font-size: var(--text-2xs);
}
</style>
