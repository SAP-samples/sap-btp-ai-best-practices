function nextAnimationFrame() {
  return new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
}

export async function afterRenderFrame(frameCount = 1) {
  for (let index = 0; index < frameCount; index += 1) {
    await nextAnimationFrame();
  }
}
