<template>
  <div class="slidev-layout title">
    <header>
      <slot />
    </header>
    <div class="left">
      <slot name="left" />
    </div>
    <div class="right">
      <slot name="right" />
    </div>
  </div>
</template>

<style scoped>
.title {
  display: grid;
  grid-template-columns: repeat(var(--grid-columns), minmax(0, 1fr));
  align-content: center;
  column-gap: var(--grid-gutter);
  height: 100%;
}

header {
  grid-column: 1 / 13;
  margin-bottom: var(--space-5);
}

.left {
  grid-column: 1 / 6;
}

.right {
  grid-column: 6 / 12;
}

:slotted(h1) {
  font-size: var(--text-5xl);
  margin-bottom: 0;
}

:slotted(h3) {
  font-size: var(--text-2xs);
  font-weight: 600;
  line-height: var(--leading-tight);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
  margin-bottom: var(--space-0);
}

:slotted(p + h3) {
  margin-top: var(--space-3);
}

:slotted(ul) {
  list-style: none;
  padding: 0;
}

:slotted(li),
:slotted(p) {
  font-size: var(--text-base);
  line-height: var(--leading-list);
}
</style>
