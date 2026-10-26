<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

// The study's tools by use, from the software and hardware tables of Chapter 3, laid out as rows like
// the benefits slide. Each row inks as Carlo names it, its roles in a lighter grey
const groups = [
  {
    name: 'Training',
    tools: [{ name: 'Python, PyTorch, Hugging Face Transformers', role: 'Standard for this kind of model' }],
  },
  {
    name: 'Optimizers',
    tools: [{ name: 'LibMTL', role: '11 of the 13, on tested code' }],
  },
  {
    name: 'Labeling',
    tools: [{ name: 'doccano', role: 'Categories and entity spans' }],
  },
  {
    name: 'Prototype',
    tools: [
      { name: 'React', role: 'Interface' },
      { name: 'FastAPI', role: 'Server' },
      { name: 'PostgreSQL', role: 'Database' },
    ],
  },
  {
    name: 'Testbed',
    tools: [
      { name: 'Acer Predator Helios Neo 16 laptop', role: 'All training and measurement' },
      { name: 'NVIDIA RTX 4060 with 8 GB', role: 'Graphics processor' },
    ],
  },
]

// The same motion tokens as the other slides
const enter = { duration: 0.3, ease: 'power1.out' }

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  const css = getComputedStyle(root.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  // Every row waits in a faint grey until its turn
  gsap.set('.cell', { color: colour('faint') })

  timeline.addLabel('step0')
  groups.forEach((_, index) => {
    timeline.to(`.group-${index} .cell:not(.role)`, { color: colour('ink'), ...enter })
    timeline.to(`.group-${index} .role`, { color: colour('subtle'), ...enter }, '<')
    timeline.addLabel(`step${index + 1}`)
  })
})
</script>

<template>
  <div ref="root" class="tools">
    <div class="row heading">
      <span>Use</span>
      <span>Tool</span>
      <span>Role</span>
    </div>

    <div v-for="(group, index) in groups" :key="group.name" class="row" :class="`group-${index}`">
      <b class="cell name-cell">{{ group.name }}</b>
      <ul>
        <li v-for="tool in group.tools" :key="tool.name">
          <span class="cell">{{ tool.name }}</span>
          <span class="cell role">{{ tool.role }}</span>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.tools {
  display: grid;
  align-content: center;
  height: 456px;
}

.row,
li {
  display: grid;
  gap: var(--space-4);
}

.row {
  grid-template-columns: 180px 1fr;
  padding: 1rem 0;
  font-size: var(--text-xs);
  line-height: var(--leading-list);
}

/* A tool and its role share the row's two right-hand columns */
li {
  grid-template-columns: 1.3fr 1fr;
}

.row + .row {
  border-top: 1px solid var(--color-faint);
}

.heading {
  grid-template-columns: 180px 1.3fr 1fr;
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
