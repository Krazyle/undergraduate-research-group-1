-- Pandoc writes "[TABLE]" for tables that GFM pipe tables cannot hold
-- (lists, paragraphs, or merged cells). Keep those tables as raw HTML instead.
function Table(tbl)
  local doc = pandoc.Pandoc({ tbl })
  if pandoc.write(doc, "gfm-raw_html"):match("%[TABLE%]") then
    return pandoc.RawBlock("gfm", pandoc.write(doc, "html"))
  end
end
