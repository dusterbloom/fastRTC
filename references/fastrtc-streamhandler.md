Stream Handlers
These abstract base classes define the core interfaces for handling audio and video streams within FastRTC. Concrete handlers like ReplyOnPause inherit from these.

StreamHandlerBase Class

StreamHandlerBase(
    expected_layout: Literal["mono", "stereo"] = "mono",
    output_sample_rate: int = 24000,
    output_frame_size: int | None = None,  # Deprecated
    input_sample_rate: int = 48000,
)
Base class for handling media streams in FastRTC.

Provides common attributes and methods for managing stream state, communication channels, and basic configuration. This class is intended to be subclassed by concrete stream handlers like StreamHandler or AsyncStreamHandler.

Attributes
Name	Type	Description
expected_layout	Literal["mono", "stereo"]	The expected channel layout of the input audio ('mono' or 'stereo').
output_sample_rate	int	The target sample rate for the output audio.
output_frame_size	int	The desired number of samples per output audio frame.
input_sample_rate	int	The expected sample rate of the input audio.
channel	DataChannel \| None	The WebRTC data channel for communication.
channel_set	asyncio.Event	Event indicating if the data channel is set.
args_set	asyncio.Event	Event indicating if additional arguments are set.
latest_args	str \| list[Any]	Stores the latest arguments received.
loop	asyncio.AbstractEventLoop	The asyncio event loop.
phone_mode	bool	Flag indicating if operating in telephone mode.
Methods
__init__

__init__(
    expected_layout: Literal["mono", "stereo"] = "mono",
    output_sample_rate: int = 24000,
    output_frame_size: int | None = None,  # Deprecated
    input_sample_rate: int = 48000,
)
Initializes the StreamHandlerBase.

Args:

Name	Type	Description
expected_layout	Literal["mono", "stereo"]	Expected input audio layout ('mono' or 'stereo').
output_sample_rate	int	Target output audio sample rate.
output_frame_size	int \| None	Deprecated. Frame size is now derived from sample rate.
input_sample_rate	int	Expected input audio sample rate.
clear_queue

clear_queue()
Clears the internal processing queue via the registered callback.

send_message

async send_message(msg: str)
Asynchronously sends a message over the data channel.

Args:

Name	Type	Description
msg	str	The string message to send.
send_message_sync

send_message_sync(msg: str)
Synchronously sends a message over the data channel. Runs the async send_message in the event loop and waits for completion.

Args:

Name	Type	Description
msg	str	The string message to send.
reset

reset()
Resets the argument set event.

shutdown

shutdown()
Placeholder for shutdown logic. Subclasses can override.

StreamHandler Class

StreamHandler(
    expected_layout: Literal["mono", "stereo"] = "mono",
    output_sample_rate: int = 24000,
    output_frame_size: int | None = None,  # Deprecated
    input_sample_rate: int = 48000,
)
Abstract base class for synchronous stream handlers.

Inherits from StreamHandlerBase and defines the core synchronous interface for processing audio streams. Subclasses must implement receive, emit, and copy.

(Inherits Attributes and Methods from StreamHandlerBase)

Abstract Methods
receive

@abstractmethod
receive(frame: tuple[int, npt.NDArray[np.int16]]) -> None
Process an incoming audio frame synchronously.

Args:

Name	Type	Description
frame	tuple[int, npt.NDArray[np.int16]]	A tuple containing the sample rate (int) and the audio data as a numpy array (int16).
emit

@abstractmethod
emit() -> EmitType
Produce the next output chunk synchronously. This method is called repeatedly to generate the output to be sent back over the stream.

Returns:

Type	Description
EmitType	An output item conforming to EmitType, which could be audio data, additional outputs, control signals (like CloseStream), or None.
copy

@abstractmethod
copy() -> StreamHandler
Create a copy of this synchronous stream handler instance. Used to create a new handler for each connection.

Returns:

Type	Description
StreamHandler	A new instance of the concrete StreamHandler subclass.
start_up

start_up()
Optional synchronous startup logic.

AsyncStreamHandler Class

AsyncStreamHandler(
    expected_layout: Literal["mono", "stereo"] = "mono",
    output_sample_rate: int = 24000,
    output_frame_size: int | None = None,  # Deprecated
    input_sample_rate: int = 48000,
)
Abstract base class for asynchronous stream handlers.

Inherits from StreamHandlerBase and defines the core asynchronous interface using coroutines (async def) for processing audio streams. Subclasses must implement receive, emit, and copy. The start_up method must also be a coroutine.

(Inherits Attributes and Methods from StreamHandlerBase)

Abstract Methods
receive

@abstractmethod
async receive(frame: tuple[int, npt.NDArray[np.int16]]) -> None
Process an incoming audio frame asynchronously.

Args:

Name	Type	Description
frame	tuple[int, npt.NDArray[np.int16]]	A tuple containing the sample rate (int) and the audio data as a numpy array (int16).
emit

@abstractmethod
async emit() -> EmitType
Produce the next output chunk asynchronously. This coroutine is called to generate the output to be sent back over the stream.

Returns:

Type	Description
EmitType	An output item conforming to EmitType, which could be audio data, additional outputs, control signals (like CloseStream), or None.
copy

@abstractmethod
copy() -> AsyncStreamHandler
Create a copy of this asynchronous stream handler instance. Used to create a new handler for each connection.

Returns:

Type	Description
AsyncStreamHandler	A new instance of the concrete AsyncStreamHandler subclass.
start_up

async start_up()
Optional asynchronous startup logic. Must be a coroutine (async def).

