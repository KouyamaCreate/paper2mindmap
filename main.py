import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import argparse
import tkinter as tk
from tkinter import filedialog
import subprocess # 追加
import webbrowser # 追加
import tempfile # 追加
import shutil # 追加

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIキーのチェック
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI APIキーが設定されていません。.envファイルを確認してください。")

# OpenAIクライアントの初期化
client = OpenAI(api_key=api_key)

def extract_text_from_pdf(file_path: str, max_pages: int = 11) -> list[str]:
    """
    PDFファイルからテキストを抽出し、指定されたチャンクサイズで分割する。

    Args:
        file_path (str): PDFファイルのパス。
        max_pages (int): 読み込む最大ページ数。

    Returns:
        list[str]: 分割されたテキストチャンクのリスト。
    """
    if not file_path: # ファイルパスが空の場合のエラーハンドリングを追加
        print("エラー: ファイルパスが指定されていません。")
        return []
    print(f"'{file_path}' からテキストを抽出中...")
    try:
        loader = PyPDFLoader(file_path)
        # ページ数を制限してロード
        pages = loader.load()
        if len(pages) > max_pages:
            print(f"警告: PDFのページ数が{max_pages}を超えています。最初の{max_pages}ページのみ処理します。")
            pages = pages[:max_pages]

        if not pages:
            print("エラー: PDFからページを読み込めませんでした。")
            return []

        # テキスト分割ツールの初期化 (ユーザー指定の設定を使用)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="p50k_base", # GPT-3向けエンコーディング。GPT-4o miniには cl100k_base が適している可能性あり
            chunk_size=1000,
            chunk_overlap=100,
        )

        # ドキュメントを分割
        sub_docs = text_splitter.split_documents(pages)
        sub_texts = [doc.page_content for doc in sub_docs]
        print(f"テキスト抽出完了。{len(sub_texts)}個のチャンクに分割されました。")
        return sub_texts

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {file_path}")
        return []
    except Exception as e:
        print(f"エラー: PDF処理中に予期せぬエラーが発生しました - {e}")
        return []

def generate_mindmap_text(text_chunks: list[str]) -> str | None:
    """
    テキストチャンクを結合し、OpenAI APIに送信してマインドマップ形式のテキストを生成する。

    Args:
        text_chunks (list[str]): 抽出されたテキストチャンク。

    Returns:
        str | None: 生成されたマインドマップテキスト、またはエラーの場合はNone。
    """
    if not text_chunks:
        print("エラー: テキストチャンクが空のため、マインドマップを生成できません。")
        return None

    full_text = "\n".join(text_chunks)
    print("OpenAI APIにリクエストを送信中...")

    # プロンプトの組み立て
    system_prompt = """あなたは与えられたテキストから重要な要素を抽出し、Mermaidの**フローチャート構文 (`graph LR`)** で表現するアシスタントです。
テキストの中心となるメイントピックを特定し、それを開始ノードとします。
メイントピックから主要なサブトピックへ、さらに必要であればサブトピックから詳細へと、内容の関係性に基づいて矢印 (`-->`) で繋いでください。
各ノードは簡潔なキーワードやフレーズで表現し、ノードID（例: A, B, C）とテキスト（例: `A[概要]`, `B(詳細トピック)`）で定義してください。ノード形状は `[]` や `()` など適切に選択してください。
**応答は必ず日本語で記述し、Mermaidコードブロック（```mermaid ... ```）で囲んでください。**

例:
```mermaid
graph LR
    A[メイントピック] --> B(サブトピック1);
    A --> C(サブトピック2);
    B --> B1[詳細1];
    B --> B2[詳細2];
    C --> C1[詳細A];
```"""
    user_prompt = f"以下のテキストからMermaidのフローチャート構文 (`graph LR`) を作成してください:\n\n---\n{full_text}\n---"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # より新しいモデルに変更 (gpt-4.1-mini は存在しない可能性)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5, # 創造性を少し抑え、テキストに忠実にする
            max_tokens=1500 # 出力トークン数の上限 (必要に応じて調整)
        )
        mindmap_text = response.choices[0].message.content
        if mindmap_text:
            print("マインドマップテキストの生成完了。")
            return mindmap_text.strip()
        else:
            print("エラー: OpenAI APIから空の応答が返されました。")
            return None
    except Exception as e:
        print(f"エラー: OpenAI API呼び出し中にエラーが発生しました - {e}")
        return None

def generate_and_open_mermaid_mindmap(mermaid_text: str, output_html_path: str):
    """
    生成されたMermaidマインドマップテキストをHTMLファイルに埋め込み、ブラウザで開く。

    Args:
        mermaid_text (str): Mermaid形式のマインドマップテキスト (```mermaid ... ``` を含む)。
        output_html_path (str): 出力するHTMLファイルのパス。
    """
    # Mermaidテキストからコードブロックを除去 (```mermaid ... ``` の中身だけ取り出す)
    start_marker = "```mermaid"
    end_marker = "```"
    start_index = mermaid_text.find(start_marker)
    end_index = mermaid_text.rfind(end_marker)

    if start_index != -1 and end_index != -1 and start_index < end_index:
        mermaid_code = mermaid_text[start_index + len(start_marker):end_index].strip() # Use mermaid_code directly
    else:
        print("エラー: Mermaidコードブロックが見つかりませんでした。処理を中断します。")
        return # エラー発生時はここで関数を抜ける

    # HTMLテンプレート
    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mermaid Flowchart</title> <!-- タイトル変更 -->
    <style>
        /* 基本的なbodyスタイルとコンテナスタイル */
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #ffffff;
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        .mermaid {{
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 80vh;
            font-size: 13px; /* 基本フォントサイズ */
        }}
        /* フローチャート用の基本スタイル */
        .mermaid .node rect,
        .mermaid .node circle,
        .mermaid .node ellipse,
        .mermaid .node polygon {{
            fill: #f9f9f9; /* ノード背景を少しグレーに */
            stroke: #999;  /* 枠線 */
            stroke-width: 1px;
        }}
        .mermaid .edgePath path {{
            stroke: #666; /* 線の色を少し濃く */
            stroke-width: 1px;
        }}
        /* 古い指定をコメントアウト */
        .flowchart-link {{ /* ユーザー提示のセレクタとプロパティを使用 */
            marker-end: none !important;
        }}
         .mermaid .node .label {{
            color: #333; /* テキスト色 */
            font-size: 13px;
        }}
        /* マインドマップ用の .level-0 スタイルは削除 */
    </style>
</head>
<body>
    <h1>Generated Mindmap</h1>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>
"""

    try:
        # HTMLファイルに書き込む
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Mermaid HTMLを '{output_html_path}' に生成しました。")

        # 生成されたHTMLファイルをブラウザで開く
        try:
            webbrowser.open(f"file://{os.path.abspath(output_html_path)}")
            print(f"'{output_html_path}' をブラウザで開きました。")
        except Exception as e:
            print(f"エラー: ブラウザでのファイルオープン中にエラーが発生しました - {e}")

    except Exception as e:
        print(f"エラー: Mermaid HTML生成・表示中にエラーが発生しました - {e}")


def select_pdf_file() -> str:
    """
    ファイル選択ダイアログを表示し、ユーザーにPDFファイルを選択させる。

    Returns:
        str: 選択されたPDFファイルのパス。キャンセルされた場合は空文字列。
    """
    root = tk.Tk()
    root.withdraw() # メインウィンドウを表示しない
    file_path = filedialog.askopenfilename(
        title="PDFファイルを選択してください",
        filetypes=[("PDFファイル", "*.pdf"), ("すべてのファイル", "*.*")]
    )
    root.destroy() # ダイアログが閉じたらTkインスタンスを破棄
    return file_path

if __name__ == "__main__":
    # コマンドライン引数のパーサーを作成 (pdf_pathを削除)
    parser = argparse.ArgumentParser(description="PDFファイルを選択し、テキストを抽出し、マインドマップを生成します。")
    # parser.add_argument("pdf_path", help="入力するPDFファイルのパス") # 削除
    parser.add_argument("-o", "--output", default="mindmap.html", help="出力するHTMLマインドマップファイル名 (デフォルト: mindmap.html)")
    parser.add_argument("-p", "--pages", type=int, default=11, help="処理する最大ページ数 (デフォルト: 11)")

    # 引数を解析
    args = parser.parse_args()

    # ファイル選択ダイアログを表示してPDFパスを取得
    pdf_path = select_pdf_file()

    if not pdf_path:
        print("ファイルが選択されませんでした。処理を終了します。")
    else:
        print(f"選択されたファイル: {pdf_path}")
        # PDFからテキストを抽出
        extracted_chunks = extract_text_from_pdf(pdf_path, args.pages)

        if extracted_chunks:
            # マインドマップテキストを生成
            mindmap_data = generate_mindmap_text(extracted_chunks)

            if mindmap_data:
                # MermaidマインドマップHTMLを生成して開く
                generate_and_open_mermaid_mindmap(mindmap_data, args.output)
            else:
                print("Mermaidマインドマップデータの生成に失敗しました。")
        else:
            print("テキスト抽出に失敗したため、処理を終了します。")