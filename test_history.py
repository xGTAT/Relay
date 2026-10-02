"""Offline regression tests: extract production classes without API/Slack setup."""
import ast
import asyncio
import json
import unittest
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

source = ast.parse(Path(__file__).with_name('bot.py').read_text())
keep = {'SessionMemory', 'groq_text_messages', 'LLMManager'}
assigns = [n for n in source.body if isinstance(n, ast.Assign) and any(getattr(x, 'id', '') == 'REMINDER_TOOL' for x in n.targets)]
module = ast.Module(body=assigns + [n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in keep], type_ignores=[])
ns = dict(defaultdict=defaultdict, deque=deque, dataclass=dataclass, field=field, MAX_MEMORY=20, MAX_MESSAGE_LENGTH=3000, json=json, ReminderManager=object, GROQ_MODEL='mock', OPENROUTER_MODEL='mock-or', build_system_prompt=lambda: 'system', log=Mock())
exec(compile(module, 'bot.py', 'exec'), ns)

class HistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ns['memory'] = ns['SessionMemory']()
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

if __name__ == '__main__':
    unittest.main()
