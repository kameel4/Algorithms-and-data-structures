from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


def _rank_key(item):
    return (-item["success_count"], item["mean_runtime_ms"])


def _algorithm_key(name):
    if name.startswith("ga-bit"):
        return "ga-bit"
    if name.startswith("ga-real"):
        return "ga-real"
    if name.startswith("pso"):
        return "pso"
    raise ValueError(f"Unsupported candidate name: {name}")


def _pick_best_per_algorithm(items):
    grouped = {}
    for item in items:
        algorithm = _algorithm_key(item["name"])
        grouped.setdefault(algorithm, []).append(item)

    best = {}
    for algorithm, candidates in grouped.items():
        best[algorithm] = sorted(candidates, key=_rank_key)[0]
    return best


def _format_success(item):
    return f'{item["success_count"]}/50'


def _format_rate(item):
    return f'{item["success_rate"]:.3f}'


def _format_runtime(item):
    return f'{item["mean_runtime_ms"]:.3f}'


def _format_max_distance(item):
    return f'{item["mean_final_max_distance"]:.6f}'


def _config_text(item):
    params = item["params"]
    name = item["name"]
    if name.startswith("pso"):
        return (
            f'swarm={params["swarm_size"]}, iter={params["iterations"]}, '
            f'c1={params["c1"]}, c2={params["c2"]}, vmax={params["vmax_ratio"]}'
        )
    if name.startswith("ga-bit"):
        return (
            f'pop={params["pop_size"]}, gen={params["generations"]}, '
            f'mut={params["mutation_prob"]}, elite={params["elite_size"]}, '
            f'k={params["tournament_k"]}, B={params["bit_width"]}'
        )
    return (
        f'pop={params["pop_size"]}, gen={params["generations"]}, '
        f'mut={params["mutation_prob"]}, elite={params["elite_size"]}, '
        f'k={params["tournament_k"]}, sigma0={params["sigma0"]}'
    )


def _paragraph(text, *, bold=False, center=False):
    ppr = ""
    if center:
        ppr = "<w:pPr><w:jc w:val=\"center\"/></w:pPr>"

    rpr = ""
    if bold:
        rpr = "<w:rPr><w:b/></w:rPr>"

    return (
        "<w:p>"
        f"{ppr}"
        "<w:r>"
        f"{rpr}"
        f"<w:t xml:space=\"preserve\">{escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def _table_cell(text, *, bold=False):
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return (
        "<w:tc>"
        "<w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>"
        "<w:p>"
        "<w:r>"
        f"{rpr}"
        f"<w:t xml:space=\"preserve\">{escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
        "</w:tc>"
    )


def _table(rows):
    parts = [
        "<w:tbl>",
        "<w:tblPr>",
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>",
        "<w:tblBorders>",
        "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>",
        "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>",
        "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>",
        "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>",
        "<w:insideH w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"auto\"/>",
        "<w:insideV w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"auto\"/>",
        "</w:tblBorders>",
        "</w:tblPr>",
    ]

    for row_index, row in enumerate(rows):
        parts.append("<w:tr>")
        for cell in row:
            parts.append(_table_cell(cell, bold=(row_index == 0)))
        parts.append("</w:tr>")

    parts.append("</w:tbl>")
    return "".join(parts)


def _document_xml(content):
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document "
        "xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        "<w:body>"
        f"{content}"
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1134\" w:right=\"1134\" w:bottom=\"1134\" w:left=\"1134\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
        "</w:body>"
        "</w:document>"
    )


def _content_types_xml():
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "</Types>"
    )


def _rels_xml():
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/>"
        "</Relationships>"
    )


def _word_rels_xml():
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"/>"
    )


def build_summary_docx(validation_path, output_path):
    items = json.loads(Path(validation_path).read_text(encoding="utf-8"))
    best = _pick_best_per_algorithm(items)

    pso_best = best["pso"]
    ga_bit_best = best["ga-bit"]
    ga_real_best = best["ga-real"]

    ga_real_old = next(item for item in items if item["name"] == "ga-real-old-best")

    header = [
        "Алгоритм",
        "Init mode",
        "Конфигурация",
        "Успешно",
        "Success rate",
        "Среднее время, мс",
        "Mean final max distance",
    ]

    rows = [
        header,
        [
            "PSO",
            pso_best["params"]["init_mode"],
            _config_text(pso_best),
            _format_success(pso_best),
            _format_rate(pso_best),
            _format_runtime(pso_best),
            _format_max_distance(pso_best),
        ],
        [
            "GA bitwise",
            ga_bit_best["params"]["init_mode"],
            _config_text(ga_bit_best),
            _format_success(ga_bit_best),
            _format_rate(ga_bit_best),
            _format_runtime(ga_bit_best),
            _format_max_distance(ga_bit_best),
        ],
        [
            "GA real",
            ga_real_best["params"]["init_mode"],
            _config_text(ga_real_best),
            _format_success(ga_real_best),
            _format_rate(ga_real_best),
            _format_runtime(ga_real_best),
            _format_max_distance(ga_real_best),
        ],
    ]

    explanation_1 = (
        "Почему раньше побеждал real GA, а после полной оптимизации победил PSO. "
        "На раннем этапе была качественно оптимизирована только вещественная реализация GA, "
        "поэтому сравнение было несимметричным: real GA уже использовал удачную random-инициализацию "
        "и подобранные гиперпараметры, тогда как PSO и bit-GA еще не были выведены в свои лучшие режимы. "
        "Дополнительная контрольная проверка показала, что старая конфигурация real GA "
        f'({ga_real_old["params"]["init_mode"]}, {_config_text(ga_real_old)}) дает только '
        f'{_format_success(ga_real_old)} успешных запусков при среднем времени {_format_runtime(ga_real_old)} мс.'
    )

    explanation_2 = (
        "После полной оптимизации для каждого алгоритма отдельно были подобраны init_mode и ключевые параметры. "
        "Наибольший выигрыш получил PSO: для него grid-инициализация и конфигурация "
        f'{_config_text(pso_best)} обеспечили {_format_success(pso_best)} успешных запусков '
        f'при среднем времени {_format_runtime(pso_best)} мс. '
        f'Оптимизированный bit-GA тоже стал сильнее и достиг {_format_success(ga_bit_best)}, '
        f'но остался заметно медленнее. Оптимизированный real GA улучшился до {_format_success(ga_real_best)}, '
        "но по надежности и по wall-clock времени уступил PSO. "
        "Поэтому при честном сравнении лучших конфигураций всех алгоритмов победил именно PSO."
    )

    content = "".join(
        [
            _paragraph("Итоговые результаты benchmark campaign", bold=True, center=True),
            _paragraph(
                "Таблица содержит лучшие подтвержденные конфигурации каждого алгоритма "
                "после многоэтапной кампании и дополнительной 50-seed валидации."
            ),
            _table(rows),
            _paragraph(""),
            _paragraph(explanation_1),
            _paragraph(""),
            _paragraph(explanation_2),
        ]
    )

    document_xml = _document_xml(content)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _rels_xml())
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/_rels/document.xml.rels", _word_rels_xml())

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export benchmark campaign summary to DOCX.")
    parser.add_argument("--validation", required=True, help="Path to validation_summary.json")
    parser.add_argument("--output", required=True, help="Path to output .docx file")
    args = parser.parse_args()

    output_path = build_summary_docx(args.validation, args.output)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
