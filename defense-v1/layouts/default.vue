<script setup lang="ts">
import { useSlideContext } from '@slidev/client'

// The script section a slide belongs to, so the panel can tell which part of the defense is on
const { $frontmatter } = useSlideContext()
</script>

<template>
  <div class="slidev-layout default">
    <p v-if="$frontmatter.section" class="section">{{ $frontmatter.section }}</p>
    <slot />
  </div>
</template>

<style scoped>
.default {
  position: relative;
}

/* The label sits in the band above the content, lined up with its left edge. It is lighter than
   the slides' own headings, so it reads as a running head rather than part of the content */
.section {
  position: absolute;
  top: var(--margin-section);
  left: var(--margin-x);
  margin: 0;
  font-size: var(--text-2xs);
  font-weight: 400;
  line-height: var(--leading-tight);
  color: var(--color-subtle);
}
</style>
