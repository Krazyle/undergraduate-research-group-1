<script setup lang="ts">
import { Map, NavigationControl, Popup, setWorkerUrl } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
// Vite moves the map library when it bundles it, so the library cannot find its worker beside itself
import worker from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { onMounted, onUnmounted, ref } from 'vue'
// Built by scripts/marikina_posts.py. The posts are an illustration placed around each barangay's
// centre, not real posts
import data from '../data/marikina-posts.json'

// A still mock-up of the prototype's main screen as DRRMO staff in Marikina would see it. A post goes
// on the map only when the model finds a location in it, so the side panel also counts the posts
// without one. Requests or urgent needs are blue, as on slide 3, and every other category is grey
const REQUESTS = 0

const located = data.posts.length
const unlocated = data.unlocated.reduce((sum, count) => sum + count, 0)

// Every post counts in the side panel, on the map or not, largest first with the remainder last
const summary = data.categories.map((category, index) => ({
  category,
  count: data.posts.filter(post => post[2] === index).length + data.unlocated[index],
  request: index === REQUESTS,
}))
const other = summary.pop()!
summary.sort((a, b) => b.count - a.count).push(other)
const most = Math.max(...summary.map(row => row.count))

// The map opens on Marikina itself and can be moved only within the area whose tiles are saved
const view: [[number, number], [number, number]] = [[121.074, 14.612], [121.136, 14.684]]
const saved: [[number, number], [number, number]] = [[data.bounds[0][1], data.bounds[0][0]], [data.bounds[1][1], data.bounds[1][0]]]

// The post a staff member has opened, in Concepcion Dos
const selected: [number, number] = [121.12082, 14.64059]

setWorkerUrl(worker)

const container = ref<HTMLElement>()
const card = ref<HTMLElement>()
let map: Map | undefined
let observer: ResizeObserver | undefined

onMounted(async () => {
  // The saved style names its tiles, fonts, and icons relative to the deck, and the map needs full URLs
  const base = new URL(import.meta.env.BASE_URL, location.href).href
  const style = JSON.parse((await (await fetch(`${base}map/style.json`)).text()).replaceAll('{base}', base))

  const css = getComputedStyle(container.value!)
  const colour = (name: string) => css.getPropertyValue(`--color-${name}`).trim()

  map = new Map({
    container: container.value!,
    style,
    bounds: view,
    maxBounds: saved,
    // The saved tiles stop at zoom 13, and the map draws them larger up to street level
    maxZoom: 16,
    // The arrow keys stay with Slidev for changing slides, and a city map has no need to turn or tilt
    keyboard: false,
    dragRotate: false,
    pitchWithRotate: false,
    touchPitch: false,
    attributionControl: { compact: false },
  })
  map.touchZoomRotate.disableRotation()
  map.addControl(new NavigationControl({ showCompass: false }), 'top-right')

  map.on('load', () => {
    map!.addSource('posts', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: data.posts.map(([lat, lon, category]) => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [lon, lat] },
          properties: { request: category === REQUESTS },
        })),
      },
    })
    map!.addLayer({
      id: 'posts',
      type: 'circle',
      source: 'posts',
      paint: {
        'circle-radius': 3.5,
        'circle-color': ['case', ['get', 'request'], colour('classification'), colour('ink')],
        'circle-opacity': ['case', ['get', 'request'], 0.7, 0.35],
      },
    })
    map!.addSource('selected', { type: 'geojson', data: { type: 'Point', coordinates: selected } })
    map!.addLayer({
      id: 'selected',
      type: 'circle',
      source: 'selected',
      paint: { 'circle-radius': 6, 'circle-color': colour('classification'), 'circle-stroke-color': colour('ink'), 'circle-stroke-width': 2 },
    })

    // The card opens below its dot, over the hills where fewer posts sit
    new Popup({ closeButton: false, closeOnClick: false, anchor: 'top', offset: 12, maxWidth: '250px' })
      .setLngLat(selected)
      .setDOMContent(card.value!)
      .addTo(map!)
  })

  // Slidev lays out a slide before it is shown, so the map redraws once its panel has a size
  observer = new ResizeObserver(() => map?.resize())
  observer.observe(container.value!)
})

onUnmounted(() => {
  observer?.disconnect()
  map?.remove()
})
</script>

<template>
  <div class="app">
    <section ref="container" class="map" />

    <!-- The opened post, which the map moves into its popup -->
    <div class="hidden">
      <div ref="card" class="card">
        <p>Need ng tubig at pagkain dito sa Brgy. Concepcion Dos, Marikina. Stranded kami sa 2nd floor</p>
        <span class="category">Requests or urgent needs</span>
      </div>
    </div>

    <aside>
      <h4>Posts per category</h4>
      <div v-for="row in summary" :key="row.category" class="bar-row">
        <span class="label">{{ row.category }}</span>
        <span class="bar"><span :class="{ request: row.request }" :style="{ width: `${(row.count / most) * 100}%` }" /></span>
        <span class="count">{{ row.count }}</span>
      </div>

      <div class="placed">
        <div><span>On the map</span><b>{{ located }}</b></div>
        <div><span>Without a location</span><b>{{ unlocated }}</b></div>
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

.map {
  min-width: 0;
}

.hidden {
  display: none;
}

.card {
  display: grid;
  gap: 8px;
  font-family: Inter, sans-serif;
  font-size: 13px;
  line-height: 1.35;
  color: var(--color-ink);
}

p {
  margin: 0;
  font-size: inherit;
  line-height: inherit;
}

.category {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-classification);
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
  opacity: 0.35;
}

.bar span.request {
  background: var(--color-classification);
  opacity: 0.7;
}

.count {
  text-align: right;
  color: var(--color-subtle);
}

.placed {
  display: grid;
  gap: 4px;
  margin-top: auto;
  font-size: 12px;
}

.placed div {
  display: flex;
  justify-content: space-between;
}

.placed span {
  color: var(--color-subtle);
}

.map :deep(.maplibregl-ctrl-attrib) {
  font-size: 10px;
}

/* The popup is drawn by the map outside the component's own elements, so its frame is styled through it */
.map :deep(.maplibregl-popup-content) {
  padding: 10px 12px;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgb(0 0 0 / 0.12);
}
</style>
