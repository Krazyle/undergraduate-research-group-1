import { useSlideContext } from '@slidev/client'
import gsap from 'gsap'
import { onMounted, onUnmounted, watch, type Ref } from 'vue'

// The build callback adds labels step0, step1, … to match the slide's clicks
export function useClickTimeline(scope: Ref<Element | undefined>, build: (timeline: gsap.core.Timeline) => void) {
  const { $clicks } = useSlideContext()

  let context: gsap.Context
  let timeline: gsap.core.Timeline
  let move: gsap.core.Tween | undefined

  onMounted(() => {
    context = gsap.context(() => {
      // GSAP rounds pixel values by default, which makes a line drawn with pathLength="1" jump in one step
      timeline = gsap.timeline({ paused: true, defaults: { duration: 0.5, ease: 'power2.out', autoRound: false } })
      build(timeline)
    }, scope.value)

    timeline.seek(`step${$clicks.value}`)
  })

  watch($clicks, (step, previous) => {
    move?.kill()
    if (step === previous + 1) {
      // Finish a click that is still playing, so the next one starts at once
      timeline.seek(`step${previous}`)
      move = timeline.tweenTo(`step${step}`)
    }
    else
      timeline.seek(`step${step}`)
  })

  onUnmounted(() => context.revert())
}
