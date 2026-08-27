import { useRef, useState } from 'react'

const ACCEPTED = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
const MAX_BYTES = 8 * 1024 * 1024

// Reject obvious problems here so an unusable file never costs a round trip.
// The backend repeats these checks; a browser is not a trust boundary.
export function validateFile(file) {
  if (!ACCEPTED.includes(file.type)) {
    return 'Select a JPG, PNG or WebP image.'
  }
  if (file.size > MAX_BYTES) {
    return 'Image must be smaller than 8 MB.'
  }
  return null
}

export default function ImageUpload({ file, previewUrl, onSelect, onClear, disabled }) {
  const inputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)

  function handleFiles(fileList) {
    const picked = fileList && fileList[0]
    if (picked) onSelect(picked)
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragActive(false)
    if (!disabled) handleFiles(event.dataTransfer.files)
  }

  if (file && previewUrl) {
    return (
      <div className="preview">
        <img src={previewUrl} alt="Selected leaf" />
        <div className="preview-meta">
          <strong>{file.name}</strong>
          {(file.size / 1024).toFixed(0)} KB
          <div className="actions">
            <button type="button" onClick={onClear} disabled={disabled}>
              Choose a different image
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={dragActive ? 'dropzone is-active' : 'dropzone'}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragActive(true)
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
      }}
    >
      <p>Drop a leaf image here, or click to browse</p>
      <span className="hint">JPG, PNG or WebP, up to 8 MB</span>
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept={ACCEPTED.join(',')}
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  )
}
