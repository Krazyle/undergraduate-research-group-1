<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, draw } from '../shared/motion'

const chart = { left: 262, right: 884, top: 44, row: 34 }
const month = (chart.right - chart.left) / 6
const x = (time: number) => chart.left + time * month
const months = ['Nov 2026', 'Dec', 'Jan 2027', 'Feb', 'Mar', 'Apr']

type Kind = 'documentation' | 'data' | 'coding' | 'training'
const tasks: { name: string, kind: Kind, start: number, end: number, step: number, extra?: { show: string, at: string } }[] = [
  { name: 'Revise Chapters 1 to 3', kind: 'documentation', start: 0, end: 0.5, step: 1 },
  { name: 'Collect and prepare posts', kind: 'data', start: 0.5, end: 1, step: 2 },
  { name: 'Full labeling', kind: 'data', start: 1, end: 2.75, step: 2, extra: { show: '.holiday', at: '>-0.1' } },
  { name: 'Benchmark code and tests', kind: 'coding', start: 1, end: 2.75, step: 3 },
  { name: 'Prototype, built and tested', kind: 'coding', start: 1, end: 2.75, step: 3 },
  { name: 'Train the models', kind: 'training', start: 2.75, end: 3.75, step: 4, extra: { show: '.depends', at: '<0.2' } },
  { name: 'Write Chapters 4 and 5', kind: 'documentation', start: 3, end: 4, step: 5 },
]
const rowY = (index: number) => chart.top + index * chart.row
const bottom = rowY(tasks.length + 1)

const holiday = { start: 1.65, end: 2.1 }
const defense = 5.12

const kinds: { kind: Kind, name: string }[] = [
  { kind: 'documentation', name: 'Documentation' },
  { kind: 'data', name: 'Data' },
  { kind: 'coding', name: 'Coding and testing' },
  { kind: 'training', name: 'Training' },
]

const root = ref<SVGSVGElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  gsap.set('.task, .holiday, .depends, .buffer, .finish', { opacity: 0 })
  gsap.set('.task .bar', { attr: { width: 0 } })

  timeline.addLabel('step0')

  const last = Math.max(...tasks.map(task => task.step))
  for (let step = 1; step <= last; step++) {
    timeline
      .to(`.task.step-${step}`, { opacity: 1, ...enter })
      .to(`.task.step-${step} .bar`, { attr: { width: (_: number, element: Element) => Number(element.getAttribute('data-width')) }, ...draw }, '<')
    for (const { extra } of tasks.filter(task => task.step === step && task.extra))
      timeline.to(extra!.show, { opacity: 1, ...enter }, extra!.at)
    timeline.addLabel(`step${step}`)
  }

  timeline
    .to('.finish, .buffer', { opacity: 1, ...enter })
    .to('.defense-mark', { fill: colour('positive'), ...enter }, '<')
    .to('.defense-label', { fill: colour('positive'), ...enter }, '<')
    .addLabel('step6')
})
</script>

<template>
  <svg ref="root" viewBox="0 0 884 384" width="884" height="384">
    <g v-for="(name, index) in months" :key="name">
      <text class="month" :x="x(index) + 6" y="24">{{ name }}</text>
      <path class="grid" :d="`M ${x(index)} 32 V ${bottom}`" />
    </g>
    <path class="grid" :d="`M ${chart.right} 32 V ${bottom}`" />

    <g class="buffer">
      <rect class="buffer-fill" :x="x(4)" :y="chart.top - 6" :width="month" :height="bottom - chart.top + 6" />
      <text class="buffer-label" :x="x(4.5)" :y="rowY(3) + 22" text-anchor="middle">Buffer</text>
    </g>

    <g v-for="(task, index) in tasks" :key="task.name" class="task" :class="`step-${task.step}`">
      <text class="task-name" x="0" :y="rowY(index) + 22">{{ task.name }}</text>
      <rect
        class="bar" :class="task.kind" :x="x(task.start)" :y="rowY(index) + 7"
        :width="x(task.end) - x(task.start)" :data-width="x(task.end) - x(task.start)" height="20" rx="4"
      />
    </g>

    <g class="holiday">
      <rect class="holiday-fill" :x="x(holiday.start)" :y="rowY(2) + 7" :width="x(holiday.end) - x(holiday.start)" height="20" />
      <text class="small" :x="x((holiday.start + holiday.end) / 2)" :y="rowY(2) + 4" text-anchor="middle">Christmas break</text>
    </g>

    <g class="depends">
      <path class="depends-line" :d="`M ${x(2.75)} ${rowY(2) + 27} V ${rowY(5) + 2}`" />
      <path class="depends-head" :d="`M ${x(2.75)} ${rowY(5) + 7} l -5 -9 h 10 z`" />
    </g>

    <text class="task-name defense-label" x="0" :y="rowY(tasks.length) + 22">Final defense</text>
    <path class="defense-mark" :d="`M ${x(defense)} ${rowY(tasks.length) + 5} l 12 12 l -12 12 l -12 -12 z`" />

    <g class="finish">
      <path class="finish-line" :d="`M ${x(4)} 32 V ${bottom + 8}`" />
      <text class="finish-label" :x="x(4) - 8" :y="bottom + 26" text-anchor="end">Every task ends by the end of February</text>
    </g>

    <g v-for="(item, index) in kinds" :key="item.kind" :transform="`translate(${index * 168} 372)`">
      <rect class="bar" :class="item.kind" x="0" y="-12" width="16" height="12" rx="3" />
      <text class="small" x="22" y="-1">{{ item.name }}</text>
    </g>
  </svg>
</template>

<style scoped>
.month {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-muted);
}

.grid {
  stroke: var(--color-land);
  stroke-width: 1.5;
}

.task-name {
  font-size: var(--text-2xs);
  fill: var(--color-ink);
}

.small {
  font-size: 0.85rem;
  fill: var(--color-muted);
}

.bar.documentation {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.bar.data {
  fill: var(--color-subtle);
}

.bar.coding {
  fill: var(--color-ink);
}

.bar.training {
  fill: var(--color-muted);
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-dasharray: 4 3;
}

.holiday-fill {
  fill: var(--color-surface);
  opacity: 0.55;
}

.depends-line {
  stroke: var(--color-ink);
  stroke-width: 1.5;
  stroke-dasharray: 4 3;
}

.depends-head {
  fill: var(--color-ink);
}

.buffer-fill {
  fill: var(--color-positive);
  opacity: 0.1;
}

.buffer-label {
  font-size: var(--text-xs);
  font-weight: 700;
  fill: var(--color-positive);
}

.defense-mark {
  fill: var(--color-ink);
}

.defense-label {
  font-weight: 600;
}

.finish-line {
  stroke: var(--color-ink);
  stroke-width: 2;
  stroke-dasharray: 6 4;
}

.finish-label {
  font-size: var(--text-2xs);
  font-weight: 600;
  fill: var(--color-ink);
}
</style>
