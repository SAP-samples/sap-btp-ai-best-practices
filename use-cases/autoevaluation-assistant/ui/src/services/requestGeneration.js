/**
 * Own asynchronous UI work with a monotonically increasing generation token.
 *
 * Each new request invalidates earlier owners. Mutations and route remounts can
 * call `invalidate()` without starting a request so late reads cannot commit.
 */
export class RequestGeneration {
  /**
   * Initialize an owner with no active request.
   */
  constructor() {
    this.generation = 0;
  }

  /**
   * Start a new latest-wins request and return its ownership token.
   *
   * @returns {number} Unique monotonically increasing request generation.
   */
  begin() {
    this.generation += 1;
    return this.generation;
  }

  /**
   * Invalidate every outstanding token without starting another request.
   *
   * @returns {number} New generation that no prior request owns.
   */
  invalidate() {
    this.generation += 1;
    return this.generation;
  }

  /**
   * Return whether a token still owns the latest permitted UI write.
   *
   * @param {number} token - Generation captured before asynchronous work.
   * @returns {boolean} True only for the current generation.
   */
  owns(token) {
    return token === this.generation;
  }
}
