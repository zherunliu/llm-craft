from dataclasses import dataclass, field


# 快速定义数据类，自动生成初始化方法（__init__）、字符串表示等（__repr__），相等性（__eq__）比较适合存储简单数据结构
@dataclass
class GuardrailResult:
    safe: bool = True
    # 每个实例的 failures 列表独立，不会共享，避免了可变默认参数的常见陷阱
    failures: list[str] = field(default_factory=list)


class SafeInputGuardrail:
    def __init__(self):
        self.sensitive_words = {
            "fuck",
            "shit",
            "bitch",
        }

        self.max_length = 2000

        self.dangerous_patterns = [
            "ignore previous instructions",
            "忽略之前的指令",
            "忽略上面的内容",
            "你现在是",
        ]

    def validate(self, input_text: str) -> GuardrailResult:
        result = GuardrailResult()
        input_lower = input_text.lower()

        if len(input_text) > self.max_length:
            result.safe = False
            result.failures.append(f"input too long (max {self.max_length} characters)")

        words = set(input_lower.split())
        found_sensitive = words & self.sensitive_words
        if found_sensitive:
            result.safe = False
            result.failures.append(
                f"contains sensitive words: {', '.join(found_sensitive)}"
            )

        for pattern in self.dangerous_patterns:
            if pattern.lower() in input_lower:
                result.safe = False
                result.failures.append(f"contains dangerous pattern: {pattern}")
                break

        return result


_guardrail: SafeInputGuardrail | None = None


def get_guardrail() -> SafeInputGuardrail:
    global _guardrail
    if _guardrail is None:
        _guardrail = SafeInputGuardrail()
    return _guardrail
