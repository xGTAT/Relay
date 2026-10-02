"""Offline regression tests: extract production classes without API/Slack setup."""
import ast
import asyncio
import json
import time
import unittest
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from store import Store
from pdfindex import chunk_pages, extract_pages

source = ast.parse(Path(__file__).with_name('bot.py').read_text())
keep = {'SessionMemory', 'groq_text_messages', 'LLMManager', 'ReminderManager', '_format_delay', 'format_passages', 'run_tool', '_human_delta', '_fmt_when', 'deadline_lines', 'ingest_pdfs', 'download_slack_file'}
assigns = [n for n in source.body if isinstance(n, ast.Assign) and any(getattr(x, 'id', '') in {'REMINDER_TOOL', 'ASK_PDF_TOOL', 'QUIZ_TOOL', 'ADD_DEADLINE_TOOL', 'LIST_DEADLINES_TOOL', 'TRIAGE_TOOL', 'COMPLETE_DEADLINE_TOOL', 'TOOLS'} for x in n.targets)]
module = ast.Module(body=assigns + [n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in keep], type_ignores=[])
ns = dict(defaultdict=defaultdict, deque=deque, dataclass=dataclass, field=field, MAX_MEMORY=20, MAX_MESSAGE_LENGTH=3000, json=json, asyncio=asyncio, time=time, datetime=__import__('datetime').datetime, MAX_TOOL_CHARS=6000, MAX_PDF_BYTES=25*1024*1024, extract_pages=extract_pages, chunk_pages=chunk_pages, store=Store(':memory:'), SLACK_BOT_TOKEN='x', aiohttp=Mock(), Optional=object, Store=Store, GROQ_MODEL='mock', OPENROUTER_MODEL='mock-or', build_system_prompt=lambda: 'system', log=Mock())
exec(compile(module, 'bot.py', 'exec'), ns)

class HistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ns['memory'] = ns['SessionMemory'](Store(':memory:'))
        self.manager = ns['LLMManager'].__new__(ns['LLMManager'])
        self.create = AsyncMock()
        self.manager.groq = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        self.reminders = Mock()
        self.or_create = AsyncMock()
        self.manager.openrouter = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.or_create)))

    def response(self, content, tool_calls=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])

    def test_legacy_and_openrouter_text_conversion(self):
        converted = ns['groq_text_messages']([
            {'role':'system','content':'system'},
            {'role':'user','content':'hello'},
            {'role':'model','content':'hi'},
            {'role':'assistant','content':'normal'},
            {'role':'model','parts':[{'text':'done'}, {'function_call':{'name':'x'}}]},
            {'role':'tool','content':'orphan','tool_call_id':'missing'},
            {'role':'invalid','content':'bad'},
            {'role':'model','parts':[{'function_call':{'name':'x'}}]},
        ])
        self.assertEqual(converted, [
            {'role':'system','content':'system'}, {'role':'user','content':'hello'},
            {'role':'assistant','content':'hi'}, {'role':'assistant','content':'normal'},
            {'role':'assistant','content':'done'}])

    async def test_multiturn_fallback_and_new_memory(self):
        ns['memory'].add('u', 'user', 'previous')
        ns['memory'].add('u', 'model', 'legacy reply')
        self.manager._call_openrouter = AsyncMock(side_effect=RuntimeError('503'))
        self.create.side_effect = [self.response('first'), self.response('second')]
        for text in ('next', 'again'):
            await self.manager.chat('u','name',text,self.reminders,'c')
        for call in self.create.call_args_list:
            self.assertTrue(all(m['role'] in {'system','user','assistant'} for m in call.kwargs['messages']))
        self.assertEqual([m['role'] for m in ns['memory'].get_history('u')][-4:], ['user','assistant','user','assistant'])

    async def test_reminder_tool_followup(self):
        tc = SimpleNamespace(id='call_1', function=SimpleNamespace(name='schedule_reminder', arguments='{"delay_seconds":10,"reminder_text":"test"}'))
        self.reminders.schedule.return_value = 'Reminder set'
        self.create.side_effect = [self.response(None,[tc]), self.response('Done')]
        result = await self.manager._call_groq([{'role':'model','content':'old'}, {'role':'user','content':'remind'}],self.reminders,'c','u')
        self.assertEqual(result,'Done')
        self.reminders.schedule.assert_called_once_with(channel_id='c',user_id='u',delay_seconds=10,reminder_text='test')
        messages = self.create.call_args_list[1].kwargs['messages']
        self.assertTrue(all(isinstance(m,dict) for m in messages))
        self.assertEqual(messages[-2]['role'],'assistant')
        self.assertEqual(messages[-2]['tool_calls'][0]['id'],messages[-1]['tool_call_id'])
        self.assertEqual(messages[-1]['role'],'tool')

    async def test_reminder_followup_failure_returns_confirmation(self):
        tc = SimpleNamespace(id='call_1', function=SimpleNamespace(name='schedule_reminder', arguments='{"delay_seconds":10,"reminder_text":"test"}'))
        self.reminders.schedule.return_value = 'Reminder set'
        self.create.side_effect = [self.response(None,[tc]), RuntimeError('network')]
        result = await self.manager._call_groq([{'role':'user','content':'remind'}],self.reminders,'c','u')
        self.assertEqual(result,'Reminder set')
        self.reminders.schedule.assert_called_once()

    async def test_openrouter_primary_tool_calling(self):
        tc = SimpleNamespace(id='call_9', function=SimpleNamespace(name='schedule_reminder', arguments='{"delay_seconds":60,"reminder_text":"viva prep"}'))
        self.reminders.schedule.return_value = 'Reminder set'
        self.or_create.side_effect = [self.response(None,[tc]), self.response('Sorted')]
        result = await self.manager.chat('u','name','remind me in a minute',self.reminders,'c')
        self.assertEqual(result,'Sorted')
        self.create.assert_not_called()
        first = self.or_create.call_args_list[0].kwargs
        self.assertEqual(first['model'],'mock-or')
        self.assertEqual(first['tools'][0]['function']['name'],'schedule_reminder')
        self.reminders.schedule.assert_called_once_with(channel_id='c',user_id='u',delay_seconds=60,reminder_text='viva prep')

    async def test_falls_back_to_groq_when_openrouter_fails(self):
        self.or_create.side_effect = RuntimeError('429')
        self.create.side_effect = [self.response('from groq')]
        result = await self.manager.chat('u','name','hi',self.reminders,'c')
        self.assertEqual(result,'from groq')

class PersistenceTests(unittest.IsolatedAsyncioTestCase):
    def test_memory_survives_reopen_and_windows(self):
        import os, tempfile
        path = os.path.join(tempfile.mkdtemp(), 'r.db')
        mem = ns['SessionMemory'](Store(path))
        for i in range(25):
            mem.add('u', 'user', f'm{i}')
        mem2 = ns['SessionMemory'](Store(path))
        hist = mem2.get_history('u')
        self.assertEqual(len(hist), 20)
        self.assertEqual(hist[0]['content'], 'm5')
        self.assertEqual(hist[-1]['content'], 'm24')
        mem2.clear('u')
        self.assertEqual(mem2.get_history('u'), [])

    async def test_reminders_restored_after_restart(self):
        store = Store(':memory:')
        now = time.time()
        store.add_reminder('u', 'c', now + 0.2, 'soon')
        store.add_reminder('u', 'c', now - 50, 'overdue')
        done = store.add_reminder('u', 'c', now - 10, 'done already')
        store.mark_reminder(done, 'done')
        client = SimpleNamespace(chat_postMessage=AsyncMock())
        rm = ns['ReminderManager'](client, store)
        self.assertEqual(rm.restore(), 2)
        await asyncio.sleep(1.8)
        texts = sorted(c.kwargs['text'] for c in client.chat_postMessage.call_args_list)
        self.assertEqual(len(texts), 2)
        self.assertTrue(any('overdue (sent late' in t for t in texts))
        self.assertFalse(any('done already' in t for t in texts))
        self.assertEqual(store.pending_reminders(), [])

    async def test_schedule_persists_reminder(self):
        store = Store(':memory:')
        rm = ns['ReminderManager'](SimpleNamespace(chat_postMessage=AsyncMock()), store)
        msg = rm.schedule('c', 'u', 3600, 'lab record')
        self.assertIn('lab record', msg)
        pending = store.pending_reminders()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['user_id'], 'u')
        self.assertAlmostEqual(pending[0]['fire_at'] - time.time(), 3600, delta=5)

    def test_cancel_only_own_reminder(self):
        store = Store(':memory:')
        rid = store.add_reminder('u', 'c', time.time() + 99, 'x')
        self.assertFalse(store.cancel_reminder('other', rid))
        self.assertTrue(store.cancel_reminder('u', rid))
        self.assertEqual(store.list_reminders('u'), [])

def make_pdf(pages_text):
    from reportlab.pdfgen import canvas
    import io
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for txt in pages_text:
        c.drawString(72, 750, txt)
        c.showPage()
    c.save()
    return buf.getvalue()


class PdfTests(unittest.IsolatedAsyncioTestCase):
    def test_extract_and_chunk(self):
        data = make_pdf(['Photosynthesis converts light into chemical energy.', 'Mitochondria produce ATP.'])
        pages = extract_pages(data)
        self.assertEqual(len(pages), 2)
        chunks = chunk_pages(pages)
        self.assertEqual([p for p, _ in chunks], [1, 2])
        self.assertIn('ATP', chunks[1][1])

    def test_chunk_overlap_and_long_pages(self):
        chunks = chunk_pages(['word ' * 600], size=500, overlap=50)
        self.assertGreater(len(chunks), 4)
        self.assertTrue(all(len(t) <= 500 for _, t in chunks))
        self.assertEqual(chunk_pages(['   ', '']), [])

    def test_search_is_scoped_per_user_and_replaces_reupload(self):
        st = Store(':memory:')
        st.add_document('a', 'bio.pdf', 1, [(1, 'Mitochondria produce ATP in the cell')])
        st.add_document('b', 'chem.pdf', 1, [(1, 'Mitochondria is not discussed; acids and bases')])
        hits = st.search_chunks('a', 'what do mitochondria produce?')
        self.assertEqual([h['doc_name'] for h in hits], ['bio.pdf'])
        self.assertEqual(st.search_chunks('a', 'a?'), [])
        st.add_document('a', 'bio.pdf', 1, [(1, 'Ribosomes make proteins')])
        self.assertEqual(st.search_chunks('a', 'mitochondria'), [])
        self.assertEqual(len(st.list_documents('a')), 1)
        self.assertEqual(len(st.sample_chunks('a', 3)), 1)

    async def test_ingest_pdfs_and_tools(self):
        st = Store(':memory:')
        data = make_pdf(['Newton second law: force equals mass times acceleration.'])
        fetch = AsyncMock(return_value=data)
        event = {'files': [
            {'name': 'physics.pdf', 'mimetype': 'application/pdf', 'url_private_download': 'u'},
            {'name': 'pic.png', 'mimetype': 'image/png', 'url_private_download': 'v'},
        ]}
        notes = await ns['ingest_pdfs'](event, 'u', st, fetch=fetch)
        self.assertIn('Indexed `physics.pdf`', notes[0])
        self.assertIn('Skipped `pic.png`', notes[1])
        fetch.assert_awaited_once_with('u')
        out = ns['run_tool']('ask_pdf', {'question': 'what is force?'}, None, st, 'c', 'u')
        self.assertIn('physics.pdf, p.1', out)
        quiz = ns['run_tool']('quiz_from_pdf', {'num_questions': 3}, None, st, 'c', 'u')
        self.assertIn('Write 3 questions', quiz)
        self.assertIn('acceleration', quiz)
        self.assertIn('No course PDFs', ns['run_tool']('ask_pdf', {'question': 'x'}, None, st, 'c', 'nobody'))
        self.assertIn('Unknown tool', ns['run_tool']('nope', {}, None, st, 'c', 'u'))

    async def test_ingest_handles_bad_and_scanned_pdf(self):
        st = Store(':memory:')
        bad = await ns['ingest_pdfs']({'files': [{'name': 'x.pdf', 'mimetype': 'application/pdf', 'url_private_download': 'u'}]}, 'u', st, fetch=AsyncMock(return_value=b'not a pdf'))
        self.assertIn("Couldn't read", bad[0])
        blank = await ns['ingest_pdfs']({'files': [{'name': 'y.pdf', 'mimetype': 'application/pdf', 'url_private_download': 'u'}]}, 'u', st, fetch=AsyncMock(return_value=make_pdf([''])))
        self.assertIn('no selectable text', blank[0])

    async def test_model_tool_loop_uses_pdf_passages(self):
        st = Store(':memory:')
        st.add_document('u', 'notes.pdf', 1, [(1, 'The mitochondrion is the powerhouse of the cell')])
        ns['store'] = st
        manager = ns['LLMManager'].__new__(ns['LLMManager'])
        tc = SimpleNamespace(id='c1', function=SimpleNamespace(name='ask_pdf', arguments='{"question":"powerhouse of the cell"}'))
        create = AsyncMock(side_effect=[
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='The mitochondrion (notes.pdf p.1)', tool_calls=None))]),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        out = await manager._call_openai_style(client, 'm', [{'role': 'user', 'content': 'q'}], Mock(), 'c', 'u')
        self.assertIn('mitochondrion', out)
        tool_msg = create.call_args_list[1].kwargs['messages'][-1]
        self.assertEqual(tool_msg['role'], 'tool')
        self.assertIn('notes.pdf, p.1', tool_msg['content'])

class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = Store(':memory:')
        self.rm = Mock()
        self.rm.schedule_at.return_value = 7

    def call(self, name, args, user='u'):
        return ns['run_tool'](name, args, self.rm, self.st, 'c', user)

    def test_human_delta(self):
        self.assertEqual(ns['_human_delta'](3 * 86400 + 4 * 3600), 'in 3d 4h')
        self.assertEqual(ns['_human_delta'](5400), 'in 1h 30m')
        self.assertEqual(ns['_human_delta'](-7300), 'overdue by 2h 1m')

    def test_add_list_triage_complete(self):
        out = self.call('add_deadline', {'title': 'DBMS assignment', 'due_in_seconds': 2 * 86400})
        self.assertIn('Saved deadline #1', out)
        self.assertIn('Reminder set', out)
        args = self.rm.schedule_at.call_args.args
        self.assertEqual(args[0:2], ('c', 'u'))
        self.assertAlmostEqual(args[2] - time.time(), 86400, delta=5)
        self.call('add_deadline', {'title': 'OS quiz', 'due_in_seconds': 2 * 86400 + 3600, 'remind_before_seconds': 3600})
        self.call('add_deadline', {'title': 'Hackathon form', 'due_in_seconds': 10 * 86400})
        listing = self.call('list_deadlines', {'days_ahead': 7})
        self.assertIn('DBMS assignment', listing)
        self.assertIn('OS quiz', listing)
        self.assertNotIn('Hackathon form', listing)
        self.assertLess(listing.index('DBMS'), listing.index('OS quiz'))
        triage = self.call('triage_deadlines', {})
        self.assertIn('2 deadline(s) fall within 48 hours', triage)
        self.assertIn('Hackathon form', triage)
        self.assertIn('Marked done: DBMS assignment', self.call('complete_deadline', {'deadline_id': 1}))
        self.assertNotIn('DBMS', self.call('list_deadlines', {}))
        self.assertIn('No open deadline', self.call('complete_deadline', {'deadline_id': 1}))
        self.assertIn('No open deadline', self.call('complete_deadline', {'deadline_id': 2}, user='other'))

    def test_close_or_bad_due_times(self):
        out = self.call('add_deadline', {'title': 'soon', 'due_in_seconds': 3600})
        self.assertIn('No reminder set', out)
        self.rm.schedule_at.assert_not_called()
        self.assertIn('looks wrong', self.call('add_deadline', {'title': 'x', 'due_in_seconds': -5}))
        self.assertIn('No open deadlines', self.call('triage_deadlines', {}, user='nobody'))

    def test_completing_deadline_cancels_its_reminder(self):
        rid = self.st.add_reminder('u', 'c', time.time() + 500, 'x')
        did = self.st.add_deadline('u', 'c', 'Lab', time.time() + 900, rid)
        self.st.complete_deadline('u', did)
        self.assertEqual(self.st.reminder_status(rid), 'cancelled')

    async def test_cancelled_reminder_does_not_fire(self):
        st = Store(':memory:')
        rid = st.add_reminder('u', 'c', time.time() + 0.1, 'nope')
        st.cancel_reminder('u', rid)
        client = SimpleNamespace(chat_postMessage=AsyncMock())
        rm = ns['ReminderManager'](client, st)
        rm._spawn(rid, 'c', 'u', 0.1, 'nope')
        await asyncio.sleep(0.8)
        client.chat_postMessage.assert_not_called()

    async def test_schedule_at_has_no_seven_day_clamp(self):
        st = Store(':memory:')
        rm = ns['ReminderManager'](SimpleNamespace(chat_postMessage=AsyncMock()), st)
        fire_at = time.time() + 20 * 86400
        rid = rm.schedule_at('c', 'u', fire_at, 'far')
        self.assertEqual(st.pending_reminders()[0]['fire_at'], fire_at)
        self.assertEqual(st.pending_reminders()[0]['id'], rid)

class PersonaTests(unittest.TestCase):
    def test_prompt_mentions_tools_and_keeps_rules(self):
        src = Path(__file__).with_name('bot.py').read_text()
        start = src.index('SYSTEM_PROMPT = """')
        prompt = src[start:src.index('"""', start + 20)]
        for needle in ('add_deadline', 'list_deadlines', 'triage_deadlines', 'complete_deadline',
                       'schedule_reminder', 'ask_pdf', 'quiz_from_pdf', '{max_len}',
                       'Confidentiality of internals', 'Current date & time'):
            self.assertIn(needle, prompt)

    def test_every_tool_named_in_prompt_exists(self):
        names = {t['function']['name'] for t in ns['TOOLS']}
        self.assertEqual(names, {'schedule_reminder', 'ask_pdf', 'quiz_from_pdf', 'add_deadline',
                                 'list_deadlines', 'triage_deadlines', 'complete_deadline'})

if __name__ == '__main__':
    unittest.main()
