<script setup lang="ts">
import gsap from 'gsap'
import { ref } from 'vue'
import { useClickTimeline } from '../composables/useClickTimeline'
import { enter, exit, draw, pop } from '../shared/motion'

const lines = [['Baha', 'sa', 'Marikina,'], ['need ng tubig', 'para', 'sa'], ['mga bata!']]
const real = new Set(['Marikina,', 'need ng tubig', 'mga bata!'])
const answers = new Set(['Baha', 'Marikina,', 'need ng tubig'])

const dim = 0.35

const lost = 14

const phases = [
  {
    key: 'training', x: 210, input: 'Training posts', model: 'Model learns',
    measures: [
      { key: 'time', y: 250, name: 'Training time', line: 'How long the model takes to learn' },
      { key: 'memory', y: 380, name: 'Peak memory during training', line: 'The most memory it uses to learn' },
    ],
  },
  {
    key: 'labeling', x: 674, input: 'A new post', model: 'Saved model labels it',
    measures: [
      { key: 'time', y: 250, name: 'Time to label each post', line: 'How long it takes for one new post' },
      { key: 'memory', y: 380, name: 'Size of the saved model', line: 'The storage space it takes up' },
    ],
  },
]

const root = ref<HTMLElement>()

useClickTimeline(root, (timeline) => {
  gsap.set('.model-side, .count-row, .check, .not-found, .marks-legend, .scene.transfer, .scene.cost, .diagram, .measure, .transfer-part', { opacity: 0 })
  gsap.set('.check', { scale: 0.4, transformOrigin: '50% 50%' })
  gsap.set('.multi', { width: '100%' })
  gsap.set('.drop, .dot-value', { opacity: 0 })
  gsap.set('.dot-wrap', { y: 0 })
  gsap.set('.fall', { scaleY: 0, transformOrigin: '50% 0%' })

  timeline
    .addLabel('step0')
    .to('.key-side', { opacity: dim, ...exit })
    .to('.model-side', { opacity: 1, ...enter }, '<')
    .to('.count-row.correct, .marks-legend', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step1')
    .to('.model-side .answers', { opacity: dim, ...exit })
    .to('.key-side', { opacity: 1, ...enter }, '<')
    .to('.check', { opacity: 1, scale: 1, ...pop, stagger: 0.15 }, '>')
    .to('.not-found', { opacity: 1, ...enter }, '>0.1')
    .to('.count-row.found', { opacity: 1, ...enter }, '>0.1')
    .to('.count-row.f1', { opacity: 1, ...enter }, '>0.3')
    .addLabel('step2')
    .to('.scene.f1', { opacity: 0, ...exit })
    .to('.scene.transfer', { opacity: 1, ...enter })
    .to('.transfer-part.bars', { opacity: 1, ...enter }, '<0.1')
    .to('.multi', { width: `${100 - lost}%`, ...draw })
    .to('.drop', { opacity: 1, ...enter }, '<0.2')
    .to('.transfer-part.axis', { opacity: 1, ...enter }, '<')
    .to('.dot-wrap', { y: 38, ...draw }, '<0.2')
    .to('.fall', { scaleY: 1, ...draw }, '<')
    .to('.dot-value', { opacity: 1, ...enter }, '>-0.1')
    .to('.transfer-part.result', { opacity: 1, ...enter }, '<0.2')
    .addLabel('step3')
    .to('.scene.transfer', { opacity: 0, ...exit })
    .to('.scene.cost', { opacity: 1, ...enter })
    .to('.diagram.training', { opacity: 1, ...enter }, '<')
    .to('.measure.time.training', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step4')
    .to('.diagram.labeling', { opacity: 1, ...enter })
    .to('.measure.time.labeling', { opacity: 1, ...enter }, '>-0.1')
    .addLabel('step5')
    .to('.measure.memory.training', { opacity: 1, ...enter })
    .addLabel('step6')
    .to('.measure.memory.labeling', { opacity: 1, ...enter })
    .addLabel('step7')
})
</script>

<template>
  <div ref="root" class="measures">
    <section class="scene f1">
      <div class="tag">Accuracy, the F1-score</div>
      <div class="sides">
        <div class="side key-side">
          <div class="tag">Answer key</div>
          <div class="post-card">
            <p v-for="(line, row) in lines" :key="row" class="post-line">
              <span v-for="piece in line" :key="piece" class="piece">
                <span v-if="real.has(piece)" class="box right" />
                <svg v-if="real.has(piece) && answers.has(piece)" class="check" viewBox="-11 -11 22 22" width="22" height="22">
                  <circle r="10" />
                  <path d="M -4.5 0 L -1 3.5 L 5 -4" />
                </svg>
                {{ piece }}
              </span>
              <span v-if="line.some(piece => real.has(piece) && !answers.has(piece))" class="not-found">Not found by the model</span>
            </p>
          </div>
        </div>

        <div class="side model-side">
          <div class="answers">
            <div class="tag">Model's answers</div>
            <div class="post-card">
              <p v-for="(line, row) in lines" :key="row" class="post-line">
                <span v-for="piece in line" :key="piece" class="piece">
                  <span v-if="answers.has(piece)" class="box" :class="real.has(piece) ? 'right' : 'wrong'" />
                  {{ piece }}
                </span>
              </p>
            </div>
          </div>
          <div class="counts">
            <div class="count-row correct"><b>Most are right</b><span>Of its answers, one is wrong</span></div>
            <div class="count-row found"><b>Most are found</b><span>Of the correct answers, one is missed</span></div>
            <div class="count-row f1"><b>F1-score</b><span>Balances the two</span></div>
          </div>
        </div>
      </div>
      <div class="marks-legend"><span><i class="key right" />Correct</span><span><i class="key wrong" />Wrong</span></div>
    </section>

    <section class="scene transfer">
      <div class="tag">Accuracy, negative transfer</div>
      <p class="statement">Negative transfer is the multi-task score minus the single-task score</p>
      <div class="transfer-layout">
        <div class="transfer-part bars">
          <div class="pair">
            <span class="who-label">Single-task model</span>
            <div class="long single" />
          </div>
          <div class="pair">
            <span class="who-label">Multi-task model</span>
            <div class="long track">
              <div class="multi" />
              <div class="drop" :style="{ width: `${lost}%` }"><span class="drop-label">Lost accuracy</span></div>
            </div>
          </div>
        </div>
        <div class="transfer-part axis">
          <span class="axis-label gain">Above the line, a gain</span>
          <div class="axis-line" />
          <div class="zero"><span>No change</span></div>
          <div class="fall" />
          <div class="dot-wrap">
            <div class="dot" />
            <span class="dot-value">Lower</span>
          </div>
          <span class="axis-label loss">Below the line, a loss</span>
        </div>
      </div>
      <p class="transfer-part result">The multi-task model scored lower, so this is <strong class="negative">negative transfer</strong></p>
    </section>

    <section class="scene cost">
      <svg viewBox="0 0 884 456" width="884" height="456">
        <path class="divider" d="M 442 0 V 456" />
        <g v-for="phase in phases" :key="phase.key">
          <g class="diagram" :class="phase.key">
            <text class="input-text" :x="phase.x" y="28" text-anchor="middle">{{ phase.input }}</text>
            <path class="line" :d="`M ${phase.x} 42 V 84`" />
            <path class="head" :d="`M ${phase.x} 96 l -7 -12 h 14 z`" />
            <rect class="model-box" :x="phase.x - 135" y="100" width="270" height="64" rx="12" />
            <text class="model-text" :x="phase.x" y="140" text-anchor="middle">{{ phase.model }}</text>
          </g>
          <g v-for="item in phase.measures" :key="item.name" class="measure" :class="[item.key, phase.key]">
            <text class="measure-name" :x="phase.x" :y="item.y" text-anchor="middle">{{ item.name }}</text>
            <text class="measure-line" :x="phase.x" :y="item.y + 34" text-anchor="middle">{{ item.line }}</text>
          </g>
        </g>
      </svg>
    </section>

  </div>
</template>

<style scoped>
.measures {
  position: relative;
  height: 456px;
}

.scene {
  position: absolute;
  inset: 0;
}

.tag {
  font-size: var(--text-2xs);
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
  margin-bottom: var(--space-1);
}

.sides {
  display: grid;
  grid-template-columns: 380px 380px;
  justify-content: space-between;
  margin-top: var(--space-2);
}

.post-card {
  padding: 14px 20px;
  border: 2px solid var(--color-faint);
  border-radius: 12px;
}

.post-line {
  margin: 0;
  font-size: var(--text-sm);
  line-height: 2.3;
}

.piece {
  position: relative;
  isolation: isolate;
  display: inline-block;
  padding: 0 7px;
  margin-right: 2px;
  line-height: 1.6;
}

.box {
  position: absolute;
  inset: 0;
  border-radius: 5px;
  z-index: -1;
}

.box.right,
.key.right {
  background: color-mix(in srgb, var(--color-ner) 22%, transparent);
  border: 2px solid var(--color-ner);
}

.box.wrong,
.key.wrong {
  background: color-mix(in srgb, var(--color-negative) 14%, transparent);
  border: 2px solid var(--color-negative);
}

.check {
  position: absolute;
  right: -11px;
  top: -11px;
}

.check circle {
  fill: var(--color-positive);
}

.check path {
  fill: none;
  stroke: var(--color-surface);
  stroke-width: 2.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.not-found {
  margin-left: var(--space-1);
  font-size: var(--text-2xs);
  font-weight: 600;
  color: var(--color-subtle);
}

.counts {
  margin-top: var(--space-1);
}

.count-row {
  display: flex;
  flex-direction: column;
  padding: 2px 0;
}

.count-row b {
  font-size: var(--text-lg);
  font-weight: 700;
  line-height: var(--leading-tight);
  letter-spacing: -0.01em;
}

.count-row span {
  font-size: var(--text-2xs);
  font-weight: 600;
  color: var(--color-muted);
}

.count-row.f1 {
  flex-direction: row;
  align-items: baseline;
  gap: var(--space-1);
  margin-top: 4px;
  padding-top: 6px;
  border-top: 2px solid var(--color-land);
}

.count-row.f1 b {
  white-space: nowrap;
}

.count-row.f1 span {
  color: var(--color-muted);
}

.marks-legend {
  position: absolute;
  left: 0;
  bottom: 0;
  display: flex;
  gap: var(--space-3);
  font-size: var(--text-2xs);
  color: var(--color-muted);
}

.key {
  display: inline-block;
  width: 16px;
  height: 16px;
  border-radius: 3px;
  margin-right: 8px;
  vertical-align: -2px;
}

.statement {
  font-size: var(--text-md);
  font-weight: 600;
  margin: var(--space-1) 0 var(--space-4);
}

.transfer-layout {
  display: grid;
  grid-template-columns: 1fr 200px;
  column-gap: var(--space-5);
  align-items: center;
}

.pair {
  display: grid;
  grid-template-columns: 190px 1fr;
  align-items: center;
  margin-bottom: var(--space-2);
  font-size: var(--text-xs);
}

.long {
  height: 28px;
  border-radius: 4px;
}

.single {
  width: 100%;
  background: var(--color-subtle);
}

.track {
  position: relative;
  width: 100%;
}

.multi {
  height: 100%;
  border-radius: 4px;
  background: var(--color-ink);
}

.drop {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  box-sizing: border-box;
  border: 2px dashed var(--color-negative);
  border-radius: 4px;
  background: color-mix(in srgb, var(--color-negative) 8%, var(--color-surface));
}

.drop-label {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  white-space: nowrap;
  font-size: var(--text-2xs);
  font-weight: 700;
  color: var(--color-ink);
}

.axis {
  position: relative;
  height: 160px;
}

.axis-line {
  position: absolute;
  left: 50%;
  top: 28px;
  bottom: 28px;
  border-left: 2px solid var(--color-ink);
}

.zero {
  position: absolute;
  left: calc(50% - 30px);
  top: 50%;
  width: 60px;
  border-top: 2px solid var(--color-ink);
}

.zero span {
  position: absolute;
  left: calc(100% + 8px);
  top: -12px;
  white-space: nowrap;
  font-size: var(--text-2xs);
  font-weight: 600;
  color: var(--color-muted);
}

.fall {
  position: absolute;
  left: calc(50% - 2px);
  top: 50%;
  width: 4px;
  height: 38px;
  background: var(--color-negative);
}

.dot-wrap {
  position: absolute;
  left: calc(50% - 8px);
  top: calc(50% - 8px);
}

.dot {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--color-negative);
}

.dot-value {
  position: absolute;
  left: 24px;
  top: -4px;
  white-space: nowrap;
  font-size: var(--text-xs);
  font-weight: 700;
  color: var(--color-ink);
}

.axis-label {
  position: absolute;
  left: 0;
  right: 0;
  text-align: center;
  font-size: var(--text-2xs);
  color: var(--color-muted);
}

.axis-label.gain {
  top: 0;
}

.axis-label.loss {
  bottom: 0;
}

.transfer-part.result {
  margin-top: var(--space-3);
  font-size: var(--text-md);
  font-weight: 700;
  color: var(--color-ink);
}

.transfer-part.result .negative {
  color: var(--color-negative);
}

.divider {
  stroke: var(--color-land);
  stroke-width: 2;
}

.input-text {
  font-size: var(--text-xs);
  fill: var(--color-ink);
}

.model-box {
  fill: var(--color-surface);
  stroke: var(--color-ink);
  stroke-width: 2;
}

.model-text {
  font-size: var(--text-xs);
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

.measure-name {
  font-size: var(--text-md);
  font-weight: 700;
  fill: var(--color-ink);
}

.measure-line {
  font-size: var(--text-2xs);
  fill: var(--color-muted);
}
</style>
