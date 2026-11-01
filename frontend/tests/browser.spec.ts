import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/auth/me', route =>
    route.fulfill({ status: 401, contentType: 'application/json', body: JSON.stringify({ detail: 'Sign in' }) }),
  )
  await page.route('**/api/v1/auth/methods', route =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ email: false, google: false, github: true }),
    }),
  )
})

test('invite-only sign in is keyboard accessible and has no axe violations', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  const signIn = page.getByRole('link', { name: 'Continue with GitHub' })
  await expect(signIn).toBeVisible()
  await page.keyboard.press('Tab')
  await expect(signIn).toBeFocused()

  const audit = await new AxeBuilder({ page }).analyze()
  expect(audit.violations).toEqual([])
})

test('layout remains within a narrow viewport and honors reduced motion', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/')
  await expect(page.getByRole('link', { name: 'Continue with GitHub' })).toBeVisible()
  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
    transition: getComputedStyle(document.querySelector('.button')!).transitionDuration,
  }))
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport)
  expect(dimensions.transition).toBe('1e-05s')
})
