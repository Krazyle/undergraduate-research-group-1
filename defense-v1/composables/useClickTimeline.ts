import { useSlideContext } from '@slidev/client'
import gsap from 'gsap'
import { onMounted, onUnmounted, watch, type Ref } from 'vue'

// The build callback adds labels step0, step1, … to match the slide's clicks. A diagram that shares
// a slide with an earlier one starts at click `first`: it holds step0 until then, and holds its last
// step once its steps run out. With `rewind`, going back one click plays that click backward at
// twice the speed instead of cutting to the step before
export function useClickTimeline(scope: Ref<Element | undefined>, build: (timeline: gsap.core.Timeline) => void, first = 0, rewind = false) {
  const { $clicks } = useSlideContext()

  let context: gsap.Context
  let timeline: gsap.core.Timeline
  let move: gsap.core.Tween | undefined
  let last = 0

  const stepAt = (clicks: number) => Math.min(Math.max(clicks - first, 0), last)

  onMounted(() => {
    context = gsap.context(() => {
      // GSAP rounds pixel values by default, which makes a line drawn with pathLength="1" jump in one step
      timeline = gsap.timeline({ paused: true, defaults: { duration: 0.5, ease: 'power2.out', autoRound: false } })
      build(timeline)
    }, scope.value)

    last = Object.keys(timeline.labels).filter(label => label.startsWith('step')).length - 1
    timeline.seek(`step${stepAt($clicks.value)}`)
  })

  watch($clicks, (clicks, previousClicks) => {
    const step = stepAt(clicks)
    const previous = stepAt(previousClicks)
    if (step === previous)
      return
    move?.kill()
    if (step === previous + 1 || (rewind && step === previous - 1)) {
      // Finish a click that is still playing, so the next one starts at once
      timeline.seek(`step${previous}`)
      move = timeline.tweenTo(`step${step}`)
      if (step < previous)
        move.timeScale(2)
    }
    else
      timeline.seek(`step${step}`)
  })

  onUnmounted(() => context.revert())
}
