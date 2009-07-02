#include "MantidKernel/Logger.h"
#include <Poco/Logger.h>
#include <Poco/LogStream.h>
#include <Poco/Message.h>
#include <Poco/Mutex.h>
#include <Poco/NullChannel.h>
#include <iostream>
#include <sstream>

namespace Mantid
{
	namespace Kernel
	{
		Logger::LoggerList* Logger::m_LoggerList = 0;
		Poco::Mutex Logger::m_ListMtx;

		/** Constructor
		* @param name The class name invoking this logger
		*/
		Logger::Logger(const std::string& name): _log(Poco::Logger::get(name)),_enabled(true)
		{  
			_name = name;
			_logStream = new Poco::LogStream(_log);
			_nullChannel = new Poco::NullChannel;
			Poco::Logger& nullLogger = Poco::Logger::get("NULL");
			nullLogger.setChannel(_nullChannel);
			_nullStream = new Poco::LogStream(nullLogger,Poco::Message::PRIO_FATAL);

		}

		///destructor
		Logger::~Logger()
		{
			delete (_logStream);
			delete (_nullStream);
			delete (_nullChannel);
		}

		/** Returns true if the log is enabled
		* 
		*  @retval true - logging is enabled
		*  @retval false - all messages are ignored.
		*/
		bool Logger::getEnabled() const
		{
			return _enabled;
		}

		/** set if the logging is enabled
		* 
		*  @param enabled  true - logging is enabled, false - all messages are ignored.
		*/
		void Logger::setEnabled(const bool enabled)
		{
			_enabled = enabled;
		}

		/** If the Logger's log level is at least PRIO_FATAL, creates a Message with 
		*  priority PRIO_FATAL and the given message text and sends it to the attached channel.
		* 
		*  @param msg The message to log.
		*/
		void Logger::fatal(const std::string& msg)
		{
			if(_enabled)
			{
				try
				{
					_log.fatal(msg); 
				} 
				catch (std::exception& e)
				{
					//failures in logging are not allowed to throw exceptions out of the logging class
					std::cerr << e.what();
				}
			}
		}

		/** If the Logger's log level is at least PRIO_ERROR, creates a Message with priority
		*  PRIO_ERROR and the given message text and sends it to the attached channel.
		* 
		*  @param msg The message to log.
		*/
		void Logger::error(const std::string& msg)
		{
			if(_enabled)
			{
				try
				{
					_log.error(msg);
				} 
				catch (std::exception& e)
				{
					//failures in logging are not allowed to throw exceptions out of the logging class
					std::cerr << e.what();
				}
			}
		}

		/** If the Logger's log level is at least PRIO_WARNING, creates a Message with 
		*  priority PRIO_WARNING and the given message text and sends it to the attached channel.
		* 
		*  @param msg The message to log.
		*/
		void Logger::warning(const std::string& msg)
		{
			if(_enabled)
			{
				try
				{
					_log.warning(msg);
				} 
				catch (std::exception& e)
				{
					//failures in logging are not allowed to throw exceptions out of the logging class
					std::cerr << e.what();
				}
			}
		}

		/** If the Logger's log level is at least PRIO_NOTICE, creates a Message with 
		*  priority PRIO_NOTICE and the given message text and sends it to the attached channel.
		* 
		*  @param msg The message to log.
		*/
		void Logger::notice(const std::string& msg)
		{
			if(_enabled)
			{
				try
				{
					_log.notice(msg);
				} 
				catch (std::exception& e)
				{
					//failures in logging are not allowed to throw exceptions out of the logging class
					std::cerr << e.what();
				}
			}
		}
		/** If the Logger's log level is at least PRIO_INFORMATION, creates a Message with 
		*  priority PRIO_INFORMATION and the given message text and sends it to the 
		*  attached channel.
		* 
		*  @param msg The message to log.
		*/
		void Logger::information(const std::string& msg)
		{
			if(_enabled)
			{
				try
				{
					_log.information(msg);
				} 
				catch (std::exception& e)
				{
					//failures in logging are not allowed to throw exceptions out of the logging class
					std::cerr << e.what();
				}
			}
		}

		/** If the Logger's log level is at least PRIO_DEBUG, creates a Message with priority
		*  PRIO_DEBUG and the given message text and sends it to the attached channel.
		* 
		*  @param msg The message to log.
		*/
		void Logger::debug(const std::string& msg)
		{
			if(_enabled)
			{
				try
				{
					_log.debug(msg);
				} 
				catch (std::exception& e)
				{
					//failures in logging are not allowed to throw exceptions out of the logging class
					std::cerr << e.what();
				}
			}
		}

		/** Logs the given message at debug level, followed by the data in buffer.
		* 
		*  The data in buffer is written in canonical hex+ASCII form:
		*  Offset (4 bytes) in hexadecimal, followed by sixteen space-separated, 
		*  two column, hexadecimal bytes, followed by the same sixteen bytes as 
		*  ASCII characters.
		*  For bytes outside the range 32 .. 127, a dot is printed.  
		*  Note all Dump messages go out at Debug message level
		* 
		*  @param msg The message to log
		*  @param buffer the binary data to log
		*  @param length The length of the binaary data to log
		*/
		void Logger::dump(const std::string& msg, const void* buffer, std::size_t length)
		{
			if(_enabled)
			{
				try
				{
					_log.dump(msg,buffer,length);
				} 
				catch (std::exception& e)
				{
					//failures in logging are not allowed to throw exceptions out of the logging class
					std::cerr << e.what();
				}
			}
		}

		/** Returns true if at least the given log level is set.
		*  @param level The logging level it is best to use the Logger::Priority enum (7=debug, 6=information, 4=warning, 3=error, 2=critical, 1=fatal)
		*/
		bool Logger::is(int level) const
		{
			bool retVal = false;
			try
			{
				retVal = _log.is(level);
			} 
			catch (std::exception& e)
			{
				//failures in logging are not allowed to throw exceptions out of the logging class
				std::cerr << e.what();
			}
			return retVal;
		}

		void Logger::setLevel(int level)
		{
			try
			{
				_log.setLevel(level);
			} 
			catch (std::exception& e)
			{
				//failures in logging are not allowed to throw exceptions out of the logging class
				std::cerr << e.what();
			}
		}

		void Logger::setLevel(const std::string& level)
		{
			try
			{
				_log.setLevel(level);
			} 
			catch (std::exception& e)
			{
				//failures in logging are not allowed to throw exceptions out of the logging class
				std::cerr << e.what();
			}
		}

		int Logger::getLevel() const
		{
			return _log.getLevel();
		}

		Poco::LogStream* Logger::getStream()
		{
			return _enabled?_logStream:_nullStream;
		}

		/** This class implements an ostream interface to the Logger for fatal messages.
		*
		* The stream's buffer appends all characters written to it
		* to a string. As soon as a CR or LF (std::endl) is written,
		* the string is sent to the Logger.
		* @returns an std::ostream reference.
		*/
		std::ostream& Logger::fatal()
		{
			return getStream()->fatal();
		}

		/** This class implements an ostream interface to the Logger for error messages.
		*
		* The stream's buffer appends all characters written to it
		* to a string. As soon as a CR or LF (std::endl) is written,
		* the string is sent to the Logger.
		* @returns an std::ostream reference.
		*/
		std::ostream& Logger::error()
		{
			return getStream()->error();
		}

		/** This class implements an ostream interface to the Logger for warning messages.
		*
		* The stream's buffer appends all characters written to it
		* to a string. As soon as a CR or LF (std::endl) is written,
		* the string is sent to the Logger.
		* @returns an std::ostream reference.
		*/
		std::ostream& Logger::warning()
		{
			return getStream()->warning();
		}

		/** This class implements an ostream interface to the Logger for notice messages.
		*
		* The stream's buffer appends all characters written to it
		* to a string. As soon as a CR or LF (std::endl) is written,
		* the string is sent to the Logger.
		* @returns an std::ostream reference.
		*/
		std::ostream& Logger::notice()
		{
			return getStream()->notice();
		}

		/** This class implements an ostream interface to the Logger for information messages.
		*
		* The stream's buffer appends all characters written to it
		* to a string. As soon as a CR or LF (std::endl) is written,
		* the string is sent to the Logger.
		* @returns an std::ostream reference.
		*/
		std::ostream& Logger::information()
		{
			return getStream()->information();
		}

		/** This class implements an ostream interface to the Logger for debug messages.
		*
		* The stream's buffer appends all characters written to it
		* to a string. As soon as a CR or LF (std::endl) is written,
		* the string is sent to the Logger.
		* @returns an std::ostream reference.
		*/
		std::ostream& Logger::debug()
		{
			return getStream()->debug();
		}

		/** releases resources and deletes this object.
		*/
		void Logger::release()
		{
			delete(this);
		}

		/** Deletes the logger and clears it from the cache.
		* 
		*  @param name The logger to destroy. 
		*/
		void Logger::destroy(Logger& logger)
		{
			Poco::Mutex::ScopedLock lock(m_ListMtx);

			if (m_LoggerList)
			{
				LoggerList::iterator it = m_LoggerList->find(&logger);
				if (it != m_LoggerList->end())
				{
					delete(*it);
					m_LoggerList->erase(it);
				}
			}
		}

		/// Shuts down the logging framework and releases all Loggers.  
		void Logger::shutdown()
		{
			Poco::Mutex::ScopedLock lock(m_ListMtx);
			try
			{
				//first release the POCO loggers
				Poco::Logger::shutdown();

				//now delete our static cache of loggers
				if (m_LoggerList)
				{
					for (LoggerList::iterator it = m_LoggerList->begin(); it != m_LoggerList->end(); ++it)
					{
						delete(*it);
					}
					delete m_LoggerList;
					m_LoggerList = 0;
				}
			} 
			catch (std::exception& e)
			{
				//failures in logging are not allowed to throw exceptions out of the logging class
				std::cerr << e.what();
			}
		}

		/** Returns a reference to the Logger with the given name.
		*  This logger is stored until in a static list until it is destroyed, released or Logger::shutdown is called.
		* 
		*  @param name The name of the logger to use - this is usually the class name. 
		*/
		Logger& Logger::get(const std::string& name)
		{
			Logger* pLogger = new Logger(name);
			if (!m_LoggerList)
				m_LoggerList = new LoggerList;
			m_LoggerList->insert(pLogger);

			return *pLogger;
		}


	} // namespace Kernel
} // Namespace Mantid

