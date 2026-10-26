<script setup lang="ts">
import { Map, NavigationControl, Popup, setWorkerUrl } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
// Vite moves the map library when it bundles it, so the library cannot find its worker beside itself
import worker from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { onMounted, onUnmounted, ref } from 'vue'
// Built by scripts/marikina_posts.py. The posts are an illustration placed around each barangay's
// centre, not real posts
import data from '../data/marikina-posts.json'

// A mock-up of the prototype's main screen as DRRMO staff in Marikina would see it. A post goes on the
// map only when the model finds a location in it, so the side panel also counts the posts without one.
// The prototype is its own app, so it has its own tokens instead of the slides' ones

// Dots on a map can sit beside any other colour, so only three categories, the ones a response team
// acts on, get their own colour and the rest share a grey. The three pass the colour-vision checks
// against the map's background for every pair, and the side panel names each colour
const SLOTS: Record<string, string> = {
  'Requests or urgent needs': 'series-1',
  'Infrastructure and utility damage': 'series-2',
  'Displaced people and evacuations': 'series-3',
}
const slot = (category: string) => SLOTS[category] ?? 'series-rest'
const swatch = (category: string) => `var(--app-${slot(category)})`

const located = data.posts.length
const unlocated = data.unlocated.reduce((sum, count) => sum + count, 0)

// Every post counts in the side panel, on the map or not, largest first with the remainder last
const summary = data.categories.map((category, index) => ({
  category,
  count: data.posts.filter(post => post[2] === index).length + data.unlocated[index],
}))
const other = summary.pop()!
summary.sort((a, b) => b.count - a.count).push(other)
const most = Math.max(...summary.map(row => row.count))

// The map opens on Marikina itself and can be moved only within the area whose tiles are saved
const view: [[number, number], [number, number]] = [[121.074, 14.612], [121.136, 14.684]]
const saved: [[number, number], [number, number]] = [[data.bounds[0][1], data.bounds[0][0]], [data.bounds[1][1], data.bounds[1][0]]]

// The post a staff member has opened, a request in Concepcion Dos
const selected: [number, number] = [121.12082, 14.64059]
const selectedCategory = 'Requests or urgent needs'

setWorkerUrl(worker)

const root = ref<HTMLElement>()
const container = ref<HTMLElement>()
const card = ref<HTMLElement>()
let map: Map | undefined
let observer: ResizeObserver | undefined

onMounted(async () => {
  // The saved style names its tiles, fonts, and icons relative to the deck, and the map needs full URLs
  const base = new URL(import.meta.env.BASE_URL, location.href).href
  const style = JSON.parse((await (await fetch(`${base}map/style.json`)).text()).replaceAll('{base}', base))

  // The map paints on a canvas, which cannot read CSS variables, so it gets the tokens' values
  const css = getComputedStyle(root.value!)
  const token = (name: string) => css.getPropertyValue(`--app-${name}`).trim()
  const colour = ['match', ['get', 'category'], ...data.categories.flatMap((category, index) => [index, token(slot(category))]), token('series-rest')]

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
          properties: { category },
        })),
      },
    })
    // Each dot has a thin ring of the surface colour, so dots that overlap stay apart
    map!.addLayer({
      id: 'posts',
      type: 'circle',
      source: 'posts',
      paint: {
        'circle-radius': 4,
        'circle-color': colour as never,
        'circle-stroke-color': token('surface'),
        'circle-stroke-width': 1,
      },
    })

    // The opened post stays a dot of its category's colour, drawn larger with a surface ring and a
    // soft halo of the same colour, and its card opens below it
    const own = token(slot(selectedCategory))
    map!.addSource('selected', { type: 'geojson', data: { type: 'Point', coordinates: selected } })
    map!.addLayer({
      id: 'selected-halo',
      type: 'circle',
      source: 'selected',
      paint: { 'circle-radius': 13, 'circle-color': own, 'circle-opacity': 0.25 },
    })
    map!.addLayer({
      id: 'selected',
      type: 'circle',
      source: 'selected',
      paint: { 'circle-radius': 7, 'circle-color': own, 'circle-stroke-color': token('surface'), 'circle-stroke-width': 2 },
    })
    new Popup({ closeButton: false, closeOnClick: false, anchor: 'top', offset: 16, maxWidth: '250px' })
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
  <div ref="root" class="app">
    <section ref="container" class="map" />

    <!-- The opened post, which the map moves into its popup -->
    <div class="hidden">
      <div ref="card" class="card">
        <p>Need ng tubig at pagkain dito sa Brgy. Concepcion Dos, Marikina. Stranded kami sa 2nd floor</p>
        <span class="category"><i :style="{ background: swatch(selectedCategory) }" />{{ selectedCategory }}</span>
      </div>
    </div>

    <aside>
      <h4>Posts per category</h4>
      <div v-for="row in summary" :key="row.category" class="bar-row">
        <span class="label"><i :style="{ background: swatch(row.category) }" />{{ row.category }}</span>
        <span class="bar"><span :style="{ width: `${(row.count / most) * 100}%`, background: swatch(row.category) }" /></span>
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
/* The prototype's own tokens. It uses the system's interface font, as a web app would, and slots 1
   to 3 of the dataviz guide's validated categorical palette */
.app {
  --app-font: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --app-text-sm: 12px;
  --app-text-md: 13px;
  --app-radius: 6px;
  --app-surface: #ffffff;
  --app-panel: #f6f7f9;
  --app-border: #d8dce1;
  --app-text: #1f2328;
  --app-text-muted: #5c6670;
  --app-track: #e6e9ed;
  --app-series-1: #2a78d6;
  --app-series-2: #eb6834;
  --app-series-3: #1baf7a;
  --app-series-rest: #9aa1a9;

  display: grid;
  grid-template-columns: 1fr 250px;
  height: 456px;
  border: 1px solid var(--app-border);
  border-radius: 8px;
  overflow: hidden;
  font-family: var(--app-font);
  font-size: var(--app-text-md);
  line-height: 1.4;
  color: var(--app-text);
  background: var(--app-surface);
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
  font-family: var(--app-font);
  font-size: var(--app-text-md);
  line-height: 1.4;
  color: var(--app-text);
}

p {
  margin: 0;
  font-size: inherit;
  line-height: inherit;
}

/* A category's colour sits in a swatch beside its name, centred on the name's first line even when
   the name wraps, and the name stays in the text colour */
.category,
.label {
  display: flex;
  align-items: flex-start;
  gap: 6px;
}

i {
  --size: 8px;
  flex: none;
  width: var(--size);
  height: var(--size);
  /* The letters sit slightly above the middle of their line, so the swatch rises by as much */
  margin-top: calc((1.4em - var(--size)) / 2 - 0.075em);
  border-radius: 50%;
}

.category {
  font-size: var(--app-text-sm);
  font-weight: 600;
}

aside {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 14px 16px;
  border-left: 1px solid var(--app-border);
  background: var(--app-panel);
}

h4 {
  margin: 0;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

.bar-row {
  display: grid;
  grid-template-columns: 1fr 30px;
  gap: 2px 8px;
  font-size: var(--app-text-sm);
}

.label {
  grid-column: 1 / -1;
}

.bar {
  height: 8px;
  align-self: center;
  border-radius: 4px;
  background: var(--app-track);
}

.bar span {
  display: block;
  height: 100%;
  border-radius: 4px;
}

.count {
  text-align: right;
  color: var(--app-text-muted);
}

.placed {
  display: grid;
  gap: 4px;
  margin-top: auto;
  font-size: var(--app-text-sm);
}

.placed div {
  display: flex;
  justify-content: space-between;
}

.placed span {
  color: var(--app-text-muted);
}

.map :deep(.maplibregl-ctrl-attrib) {
  font-family: var(--app-font);
  font-size: 10px;
}

/* The popup is drawn by the map outside the component's own elements, so its frame is styled through it */
.map :deep(.maplibregl-popup-content) {
  padding: 10px 12px;
  border-radius: var(--app-radius);
  box-shadow: 0 2px 8px rgb(0 0 0 / 0.12);
}
</style>
