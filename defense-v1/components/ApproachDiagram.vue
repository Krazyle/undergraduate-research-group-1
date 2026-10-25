<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'

const width = 884

// Each panel is drawn around x = 0 and slides into place. A pipeline is one post, one model,
// and the outputs that model gives
const panels = [
  {
    caption: ['Earlier Philippine', 'studies'],
    pipelines: [
      { x: -66, width: 116, name: 'Model', outputs: ['classification'] },
      { x: 66, width: 116, name: 'Model', outputs: ['ner'] },
    ],
  },
  {
    caption: ['Urdaneta City', 'system'],
    pipelines: [{ x: 0, width: 116, name: 'LLM', outputs: ['classification', 'ner'] }],
  },
  {
    caption: ['This study'],
    pipelines: [{ x: 0, width: 256, name: 'Pretrained encoder', outputs: ['classification', 'ner'] }],
  },
]

// A model with two outputs branches its arrow to two swatches this far either side
const spread = 34

function outputX(outputs: string[], index: number) {
  return outputs.length === 1 ? 0 : (index === 0 ? -spread : spread)
}

// Where each panel sits when the first n panels share the width evenly
function place(index: number, count: number) {
  return (width * (index + 0.5)) / count
}

// Example posts labeled the way slide 2 labels them. The brackets are placed by this exact
// wording, set in Inter at --text-base: entity brackets under their words, measured from the
// start of the post, and one classification bracket under the whole post
const examples = [
  { kind: 'english', title: 'English post', text: 'Flooding in Houston, we need water!', entities: [[125, 216], [266, 387]], end: 393 },
  { kind: 'taglish', title: 'Taglish post', text: 'Baha na sa Marikina, need ng tubig!', entities: [[125, 217], [229, 378]], end: 384 },
]

function bracket(start: number, end: number, top: number) {
  return `M ${start} ${top} V ${top + 6} H ${end} V ${top}`
}

// Once only this study is left, it and the examples get equal space at the slide's left edge,
// between them, and at the right edge. The slide is 980 wide and the diagram starts 48 in.
// The space is sized for the wider English post, so the model stays put when it appears
const modelWidth = 256
const examplesWidth = 393
const space = (980 - modelWidth - examplesWidth) / 3
const modelX = -48 + space + modelWidth / 2
const examplesX = -48 + 2 * space + modelWidth

// The Taglish post sits level with the model alone, then moves down to share the height
const taglishDrop = 63

// A 12 px dash is centred on each corner, so it bends around the corner and covers the end tick.
// The gaps stretch so whole periods fit between the two corners
function fitDashes(path: SVGPathElement) {
  const length = path.getTotalLength()
  const dash = 12
  const periods = Math.round((length - dash) / (dash + 6))
  path.style.strokeDasharray = `${dash} ${(length - dash) / periods - dash}`
}

// The same motion tokens as slides 2 and 3
const enter = { duration: 0.3, ease: 'power1.out' }
const exit = { duration: 0.2, ease: 'power1.in' }
const draw = { duration: 0.5, ease: 'power2.out' }
const moveShort = { duration: 0.5, ease: 'power2.inOut' }

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.panel, .legend, .example, .missing, .english-source', { opacity: 0 })
  panels.forEach((_, index) => gsap.set(`.panel-${index}`, { x: place(index, index + 1) }))
  gsap.utils.toArray<SVGPathElement>('.missing').forEach(fitDashes)

  timeline
    .addLabel('step0')
    .to('.panel-0, .legend', { opacity: 1, ...enter })
    .addLabel('step1')
    .to('.panel-0', { x: place(0, 2), ...moveShort })
    .to('.panel-1', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step2')
    .to('.panel-0', { x: place(0, 3), ...moveShort })
    .to('.panel-1', { x: place(1, 3), ...moveShort }, '<')
    .to('.panel-2', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step3')
    .to('.panel-0, .panel-1, .earlier-sources', { opacity: 0, ...exit })
    .to('.panel-2', { x: width / 2, ...moveShort }, '<0.1')
    .addLabel('step4')
    .to('.panel-2', { x: modelX, ...moveShort })
    .to('.taglish', { opacity: 1, ...enter }, '>-0.1')
    .fromTo('.taglish .bracket:not(.missing)', { clipPath: 'inset(0% 50% 0% 50%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.2')
    .addLabel('step5')
    // The Taglish labels turn dashed: needed, but no dataset has them yet
    .to('.taglish', { y: taglishDrop, ...moveShort })
    .to('.taglish .bracket:not(.missing)', { opacity: 0, ...exit }, '<')
    .to('.taglish .missing', { opacity: 1, ...enter }, '<0.2')
    .to('.english, .english-source', { opacity: 1, ...enter }, '<')
    .fromTo('.english .bracket', { clipPath: 'inset(0% 50% 0% 50%)' }, { clipPath: 'inset(0% 0% 0% 0%)', ...draw }, '<0.2')
    .addLabel('step6')
})
</script>

<template>
  <div ref="root" class="approach-diagram">
    <svg viewBox="0 0 884 456" width="884" height="456">
      <g v-for="(panel, index) in panels" :key="index" class="panel" :class="`panel-${index}`">
        <g v-for="pipeline in panel.pipelines" :key="pipeline.x" :transform="`translate(${pipeline.x} 0)`">
          <text class="post" y="32" text-anchor="middle">Post</text>
          <path d="M 0 46 V 128" />
          <path class="head" d="M 0 140 l -7 -12 h 14 z" />

          <rect class="model" :x="-pipeline.width / 2" y="144" :width="pipeline.width" height="64" rx="12" />
          <text class="name" y="184" text-anchor="middle">{{ pipeline.name }}</text>

          <g v-for="(output, slot) in pipeline.outputs" :key="output" :class="output">
            <path :d="`M 0 212 V 230 H ${outputX(pipeline.outputs, slot)} V 294`" />
            <path class="head" :d="`M ${outputX(pipeline.outputs, slot)} 306 l -7 -12 h 14 z`" />
            <rect class="swatch" :x="outputX(pipeline.outputs, slot) - 14" y="312" width="28" height="28" rx="4" />
          </g>
        </g>

        <text class="tag" text-anchor="middle">
          <tspan v-for="(line, row) in panel.caption" :key="row" x="0" :y="378 + row * 22">{{ line }}</tspan>
        </text>
      </g>

      <g :transform="`translate(${examplesX} 0)`">
        <g v-for="(example, row) in examples" :key="example.kind" class="example" :class="example.kind">
          <g :transform="`translate(0 ${147 + row * taglishDrop})`">
            <text class="tag" y="-40">{{ example.title }}</text>
            <text class="post">{{ example.text }}</text>
            <path v-for="[start, end] in example.entities" :key="start" class="bracket ner" :d="bracket(start, end, 9)" />
            <path class="bracket classification" :d="bracket(0, example.end, 27)" />
            <template v-if="example.kind === 'taglish'">
              <path v-for="[start, end] in example.entities" :key="start" class="bracket ner missing" :d="bracket(start, end, 9)" />
              <path class="bracket classification missing" :d="bracket(0, example.end, 27)" />
            </template>
          </g>
        </g>
      </g>

      <!-- Placed by this exact wording, set in Inter at --text-2xs, so the legend is centred -->
      <g class="legend">
        <rect class="swatch classification" x="93" y="430" width="14" height="14" rx="3" />
        <text class="tag" x="115" y="443">Humanitarian classification</text>
        <rect class="swatch ner" x="477" y="430" width="14" height="14" rx="3" />
        <text class="tag" x="499" y="443">Named entity recognition</text>
      </g>
    </svg>

    <!-- The sources follow what is on screen: the earlier studies leave with their panels -->
    <footer class="earlier-sources">Sources: Ermino et al. (2022), Somido &amp; Reyes (2023), Yute et al. (2021), Caoile &amp; Aviles (2025)</footer>
    <footer class="english-source">Source: Ali et al. (2026)</footer>
  </div>
</template>

<style scoped>
.approach-diagram {
  position: relative;
}

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
  fill: var(--color-muted);
}

path {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 2;
}

.head {
  fill: var(--color-ink);
  stroke: none;
}

.model {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.swatch {
  fill: currentColor;
}

.classification {
  color: var(--color-classification);
}

.ner {
  color: var(--color-ner);
}

.bracket {
  stroke: currentColor;
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
