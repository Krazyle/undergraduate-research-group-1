<script setup lang="ts">
// A still mock-up of the prototype's main screen as DRRMO staff would see it, following the use cases
// of Chapter 3. The feed lists the replayed posts newest first, each already labeled by the model.
// The posts are written for this slide and are not from the dataset

// Each post is split into plain text and entity spans, which are highlighted
type Part = string | { text: string, type: string }

const posts: { time: string, parts: Part[], category: string, urgent?: boolean, status?: string }[] = [
  {
    time: '10:42',
    parts: [{ text: 'Need ng tubig at pagkain', type: 'REQUEST' }, ' dito sa ', { text: 'Brgy. Malanday', type: 'LOCATION' }, ', stranded kami sa 2nd floor'],
    category: 'Requests or urgent needs',
    urgent: true,
  },
  {
    time: '10:41',
    parts: ['Nag-deploy na ng ', { text: 'rescue boats', type: 'RESCUE' }, ' ang ', { text: 'Philippine Red Cross', type: 'ORGANIZATION' }, ' sa ', { text: 'Tumana', type: 'LOCATION' }],
    category: 'Rescue, volunteering, or donation effort',
  },
  {
    time: '10:39',
    parts: [{ text: '500 evacuees', type: 'DISPLACED' }, ' na sa covered court ng ', { text: 'San Mateo', type: 'LOCATION' }, ' dahil sa ', { text: 'Bagyong Kristine', type: 'DISASTER_TYPE' }],
    category: 'Displaced people and evacuations',
  },
  {
    time: '10:38',
    parts: ['Bagsak ang tulay sa ', { text: 'Brgy. Banaba', type: 'LOCATION' }, ', hindi madaanan ng sasakyan'],
    category: 'Infrastructure and utility damage',
  },
  {
    time: '10:36',
    parts: [{ text: 'Kailangan ng gamot', type: 'REQUEST' }, ' para sa mga bata sa evacuation center sa ', { text: 'Naga', type: 'LOCATION' }],
    category: 'Requests or urgent needs',
    urgent: true,
    status: 'Forwarded',
  },
  {
    time: '10:35',
    parts: ['Ingat po kayong lahat, praying for everyone sa ', { text: 'Bicol', type: 'LOCATION' }],
    category: 'Sympathy and support',
  },
]

const summary = [
  { category: 'Requests or urgent needs', count: 38 },
  { category: 'Sympathy and support', count: 31 },
  { category: 'Rescue, volunteering, or donation effort', count: 24 },
  { category: 'Displaced people and evacuations', count: 19 },
  { category: 'Infrastructure and utility damage', count: 15 },
]
const most = Math.max(...summary.map(row => row.count))

const filters = ['All posts', 'Urgent requests', 'Forwarded']
</script>

<template>
  <div class="app">
    <header>
      <b class="name">Typhoon Post Monitor</b>
      <span class="live"><i class="dot" />Live, replaying posts on Bagyong Kristine</span>
      <nav>
        <span v-for="(filter, index) in filters" :key="filter" class="chip" :class="{ on: index === 0 }">{{ filter }}</span>
      </nav>
    </header>

    <main>
      <section class="feed">
        <div class="columns">
          <span>Time</span>
          <span>Post</span>
          <span>Category</span>
          <span>Status</span>
        </div>
        <div v-for="post in posts" :key="post.time" class="post">
          <span class="time">{{ post.time }}</span>
          <p>
            <template v-for="(part, index) in post.parts" :key="index">
              <mark v-if="typeof part !== 'string'" :title="part.type">{{ part.text }}</mark>
              <template v-else>
                {{ part }}
              </template>
            </template>
          </p>
          <span class="category" :class="{ urgent: post.urgent }">{{ post.category }}<small>▾</small></span>
          <span class="status">{{ post.status ?? 'New' }}</span>
        </div>
      </section>

      <aside>
        <h4>Posts per category</h4>
        <div v-for="row in summary" :key="row.category" class="bar-row">
          <span class="label">{{ row.category }}</span>
          <span class="bar"><span :style="{ width: `${(row.count / most) * 100}%` }" /></span>
          <span class="count">{{ row.count }}</span>
        </div>
        <div class="corrections">12 labels corrected and saved for training</div>
        <div class="buttons">
          <span class="button">Export report</span>
          <span class="button">Export corrections</span>
        </div>
      </aside>
    </main>
  </div>
</template>

<style scoped>
.app {
  display: grid;
  grid-template-rows: auto 1fr;
  height: 456px;
  border: 1px solid var(--color-faint);
  border-radius: 8px;
  overflow: hidden;
  font-size: 13px;
  line-height: 1.35;
  color: var(--color-ink);
}

header {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: 10px 14px;
  border-bottom: 1px solid var(--color-faint);
  background: var(--color-land);
}

.name {
  font-weight: 700;
}

.live {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--color-subtle);
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--color-negative);
}

nav {
  display: flex;
  gap: 6px;
  margin-left: auto;
}

.chip,
.button {
  padding: 3px 10px;
  border: 1px solid var(--color-faint);
  border-radius: 999px;
  background: var(--color-surface);
  white-space: nowrap;
}

.chip.on {
  border-color: var(--color-ink);
  background: var(--color-ink);
  color: var(--color-surface);
}

main {
  display: grid;
  grid-template-columns: 1fr 230px;
  min-height: 0;
}

.feed {
  overflow: hidden;
  /* More posts continue below the fold */
  mask-image: linear-gradient(to bottom, black 85%, transparent);
}

.columns,
.post {
  display: grid;
  grid-template-columns: 44px 1fr 150px 70px;
  gap: 12px;
  padding: 8px 14px;
}

.columns {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-subtle);
  border-bottom: 1px solid var(--color-faint);
}

.post + .post {
  border-top: 1px solid var(--color-land);
}

.time,
.status {
  color: var(--color-subtle);
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
  align-self: start;
  padding: 2px 8px;
  border: 1px solid var(--color-classification);
  border-radius: 6px;
  font-size: 12px;
  color: var(--color-classification);
}

.category small {
  margin-left: 4px;
}

.category.urgent {
  background: var(--color-classification);
  color: var(--color-surface);
}

aside {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px 14px;
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
  grid-template-columns: 1fr 26px;
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
  background: var(--color-classification);
}

.count {
  text-align: right;
  color: var(--color-subtle);
}

.corrections {
  margin-top: auto;
  padding-top: 8px;
  border-top: 1px solid var(--color-faint);
  font-size: 12px;
  color: var(--color-subtle);
}

.buttons {
  display: grid;
  gap: 6px;
}

.button {
  font-size: 12px;
  text-align: center;
}
</style>
