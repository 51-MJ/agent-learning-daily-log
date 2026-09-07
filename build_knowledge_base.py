# build_knowledge_base.py
# 功能：从 docs/ 目录读取所有 txt 文件，支持全量重建和增量更新
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from rag_utils import add_knowledge, delete_knowledge_by_source, list_knowledge_sources

DOCS_DIR = Path(__file__).parent / "docs"

def load_all_txt_files(docs_dir: Path) -> dict:
    """读取docs目录下所有txt文件，返回 {文件名: 内容} 字典"""
    file_dict = {}
    if not docs_dir.exists():
        print(f"错误：目录 {docs_dir} 不存在")
        return file_dict

    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        print(f"警告：{docs_dir} 下没有找到 txt 文件")
        return file_dict

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                file_dict[txt_file.name] = content
                print(f"  读取：{txt_file.name}（{len(content)}字符）")
            else:
                print(f"  跳过空文件：{txt_file.name}")
    return file_dict

def main():
    print("=" * 55)
    print("知识库构建工具")
    print("=" * 55)

    # 显示当前向量库已有来源
    existing = list_knowledge_sources()
    print(f"\n当前向量库已有来源：{existing if existing else '（空）'}")

    # 选择模式
    print("\n请选择构建模式：")
    print("  1 - 全量重建（删除所有旧数据，重新导入全部txt）")
    print("  2 - 增量更新（只更新有变动的文件，新增文件追加）")
    mode = input("\n输入模式编号（默认2）：").strip() or "2"

    # 读取所有txt
    print(f"\n【读取知识库文件】")
    file_dict = load_all_txt_files(DOCS_DIR)
    if not file_dict:
        print("没有可导入的内容，程序退出")
        return

    print(f"\n共读取到 {len(file_dict)} 个文件：{list(file_dict.keys())}")

    if mode == "1":
        # 全量重建：逐个删除旧来源再重新导入
        print("\n【全量重建模式】")
        for src in existing:
            delete_knowledge_by_source(src)
        for filename, content in file_dict.items():
            print(f"\n导入：{filename}")
            add_knowledge([content], source=filename)
            print(f"  -> 完成")
    else:
        # 增量更新：已存在的文件先删旧再导新，不存在的直接追加
        print("\n【增量更新模式】")
        for filename, content in file_dict.items():
            if filename in existing:
                print(f"\n更新已有文件：{filename}")
                delete_knowledge_by_source(filename)
            else:
                print(f"\n新增文件：{filename}")
            add_knowledge([content], source=filename)
            print(f"  -> 完成")

    # 完成后显示最终来源
    final_sources = list_knowledge_sources()
    print("\n" + "=" * 55)
    print("构建完成！")
    print(f"最终向量库来源：{final_sources}")
    print(f"向量库位置：{Path(__file__).parent / 'vector_db'}")
    print("=" * 55)

if __name__ == "__main__":
    main()
