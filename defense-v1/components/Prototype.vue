<script setup lang="ts">
// Built by scripts/luzon_map.py, as on slide 3. The posts are an illustration placed by population,
// not real posts
import map from '../data/luzon-map.json'

// A still mock-up of the prototype's main screen as DRRMO staff would see it. A post goes on the map
// only when the model finds a location in it, so the side panel also counts the posts without one.
// Requests or urgent needs are blue, as on slide 3, and every other category is grey
const posts = map.posts.map(([x, y, request]) => ({ x, y, request: request === 1 }))
const mapped = posts.length

// The category counts cover every post, on the map or not, and are made up to match the dots
const summary = [
  { category: 'Requests or urgent needs', count: 412, request: true },
  { category: 'Sympathy and support', count: 351 },
  { category: 'Rescue, volunteering, or donation effort', count: 298 },
  { category: 'Displaced people and evacuations', count: 221 },
  { category: 'Infrastructure and utility damage', count: 187 },
  { category: 'Other categories', count: 227 },
]
const total = summary.reduce((sum, row) => sum + row.count, 0)
const most = Math.max(...summary.map(row => row.count))

// The post a staff member has opened, at Marikina, with its entities highlighted
const selected = { x: 148.4, y: 297.9 }
type Part = string | { text: string, type: string }
const parts: Part[] = [{ text: 'Need ng tubig at pagkain', type: 'REQUEST' }, ' dito sa ', { text: 'Brgy. Malanday', type: 'LOCATION' }, ', ', { text: 'Marikina', type: 'LOCATION' }, '. Stranded kami sa 2nd floor']
</script>

<template>
  <div class="app">
    <section class="map">
      <svg viewBox="-50 0 600 504">
        <path class="land" :d="map.land" />
        <circle v-for="(post, index) in posts" :key="index" class="post" :class="{ request: post.request }" :cx="post.x" :cy="post.y" r="2.2" />
        <circle class="ring" :cx="selected.x" :cy="selected.y" r="6" />

        <!-- The opened post sits just to the right of its dot, in the map's own units -->
        <foreignObject :x="selected.x + 14" :y="selected.y - 24" width="260" height="140">
          <div class="card">
            <p>
              <template v-for="(part, index) in parts" :key="index">
                <mark v-if="typeof part !== 'string'" :title="part.type">{{ part.text }}</mark>
                <template v-else>
                  {{ part }}
                </template>
              </template>
            </p>
            <span class="category">Requests or urgent needs<small>▾</small></span>
          </div>
        </foreignObject>
      </svg>
    </section>

    <aside>
      <h4>Posts per category</h4>
      <div v-for="row in summary" :key="row.category" class="bar-row">
        <span class="label">{{ row.category }}</span>
        <span class="bar"><span :class="{ request: row.request }" :style="{ width: `${(row.count / most) * 100}%` }" /></span>
        <span class="count">{{ row.count }}</span>
      </div>

      <div class="placed">
        <div><span>On the map</span><b>{{ mapped.toLocaleString() }}</b></div>
        <div><span>Without a location</span><b>{{ (total - mapped).toLocaleString() }}</b></div>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.app {
  display: grid;
  grid-template-columns: 1fr 250px;
  height: 456px;
  border: 1px solid var(--color-faint);
  border-radius: 8px;
  overflow: hidden;
  font-size: 13px;
  line-height: 1.35;
  color: var(--color-ink);
}

/* The map fills its panel without its own height stretching the screen */
.map {
  position: relative;
  min-width: 0;
}

svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.land {
  fill: var(--color-land);
}

.post {
  fill: var(--color-ink);
  fill-opacity: 0.3;
}

.post.request {
  fill: var(--color-classification);
  fill-opacity: 0.6;
}

.ring {
  fill: none;
  stroke: var(--color-ink);
  stroke-width: 1.5;
}

.card {
  display: grid;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid var(--color-faint);
  border-radius: 6px;
  background: var(--color-surface);
  box-shadow: 0 2px 8px rgb(0 0 0 / 0.08);
  /* The map is drawn a little smaller than the screen, so the card's text is set a little larger */
  font-size: 14px;
}

p {
  margin: 0;
  font-size: inherit;
  line-height: inherit;
}

mark {
  padding: 0 2px;
  border-radius: 3px;
  background: color-mix(in srgb, var(--color-ner) 18%, transparent);
  color: inherit;
}

.category {
  justify-self: start;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 12px;
  background: var(--color-classification);
  color: var(--color-surface);
}

.category small {
  margin-left: 4px;
}

aside {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 14px 16px;
  border-left: 1px solid var(--color-faint);
}

h4 {
  margin: 0;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-subtle);
}

.bar-row {
  display: grid;
  grid-template-columns: 1fr 30px;
  gap: 2px 8px;
  font-size: 12px;
}

.label {
  grid-column: 1 / -1;
}

.bar {
  height: 8px;
  align-self: center;
  background: var(--color-land);
}

.bar span {
  display: block;
  height: 100%;
  background: var(--color-ink);
  opacity: 0.3;
}

.bar span.request {
  background: var(--color-classification);
  opacity: 0.6;
}

.count {
  text-align: right;
  color: var(--color-subtle);
}

.placed {
  display: grid;
  gap: 4px;
  margin-top: auto;
  padding-top: 10px;
  border-top: 1px solid var(--color-faint);
  font-size: 12px;
}

.placed div {
  display: flex;
  justify-content: space-between;
}

.placed span {
  color: var(--color-subtle);
}
</style>
