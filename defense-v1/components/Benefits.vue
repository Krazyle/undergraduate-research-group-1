<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The three groups the study benefits, the main one first, laid out as rows like the literature
// slide. Each benefit inks in the order Isaiah names it, and a group's name inks with its first benefit
const groups = [
  {
    name: 'LGUs and disaster offices',
    benefits: [
      'Report type and key details from one model',
      'Runs locally, without a paid LLM',
      'Trade-offs of one model against two',
    ],
  },
  {
    name: 'Local universities',
    benefits: ['Training time and memory needed to build and update the model'],
  },
  {
    name: 'Communities hit by typhoons',
    benefits: ['Urgent requests may be found sooner'],
  },
]

let next = 0
const rows = groups.map((group) => {
  const first = next
  next += group.benefits.length
  return { ...group, first }
})

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  // Every cell waits in a faint grey until its turn
  gsap.set('.cell', { color: colour('faint') })

  timeline.addLabel('step0')
  for (let index = 0; index < next; index++) {
    const row = rows.findIndex(group => group.first === index)
    timeline.to(row >= 0 ? `.benefit-${index}, .group-${row} .name-cell` : `.benefit-${index}`, { color: colour('ink'), ...enter })
    timeline.addLabel(`step${index + 1}`)
  }
})
</script>

<template>
  <div ref="root" class="benefits">
    <div class="row heading">
      <span>Group</span>
      <span>Benefit</span>
    </div>

    <div v-for="(row, index) in rows" :key="row.name" class="row" :class="`group-${index}`">
      <b class="cell name-cell">{{ row.name }}</b>
      <ul>
        <li v-for="(benefit, slot) in row.benefits" :key="benefit" class="cell" :class="`benefit-${row.first + slot}`">{{ benefit }}</li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.benefits {
  display: grid;
  align-content: center;
  height: 456px;
}

.row {
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: var(--space-4);
  padding: 1.25rem 0;
  font-size: var(--text-xs);
  line-height: var(--leading-list);
}

.row + .row {
  border-top: 1px solid var(--color-faint);
}

.heading {
  padding-top: 0;
  padding-bottom: 0.75rem;
  border-bottom: 2px solid var(--color-ink);
  font-size: var(--text-2xs);
  font-weight: 600;
  line-height: var(--leading-tight);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
}

.heading + .row {
  border-top: none;
}

ul {
  display: grid;
  gap: var(--space-1);
  margin: 0;
  padding: 0;
  list-style: none;
}

li {
  font-size: inherit;
  line-height: inherit;
}

b {
  font-weight: 600;
}
</style>
