/** Regression tests for readable, safe assistant Markdown rendering. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {formatAssistantMarkdown,messageContent} from '../src/modules/chat-markdown.js';

test('assistant markdown renders the supported report structure',()=>{
  const html=formatAssistantMarkdown('## Overview\n\n**Ready**\n\n- First\n- Second\n\n| Item | Value |\n|---|---:|\n| Selected | 314 |\n\n`FEASIBLE`');
  assert.match(html,/<h2>Overview<\/h2>/);
  assert.match(html,/<strong>Ready<\/strong>/);
  assert.match(html,/<ul>/);
  assert.match(html,/<table>/);
  assert.match(html,/<code>FEASIBLE<\/code>/);
});

test('assistant markdown permits only safe external links',()=>{
  const html=formatAssistantMarkdown('[Report](https://example.com/report) [Unsafe](javascript:alert(1))');
  assert.match(html,/href="https:\/\/example\.com\/report"/);
  assert.match(html,/target="_blank"/);
  assert.match(html,/rel="noopener noreferrer"/);
  assert.doesNotMatch(html,/javascript:/i);
});

test('assistant markdown escapes raw html and suppresses images',()=>{
  const html=formatAssistantMarkdown('<img src=x onerror=alert(1)> ![remote](https://example.com/image.png)');
  assert.doesNotMatch(html,/<img/i);
  assert.match(html,/&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.match(html,/remote/);
});

test('user and error messages remain literal text',()=>{
  assert.deepEqual(messageContent('**not bold**','user'),{mode:'text',value:'**not bold**'});
  assert.deepEqual(messageContent('<b>error</b>','assistant',{isError:true}),{mode:'text',value:'<b>error</b>'});
  assert.equal(messageContent('**bold**','assistant').mode,'html');
});
